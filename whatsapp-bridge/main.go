package main

import (
	"context"
	"database/sql"
	"flag"
	"os"
	"os/signal"
	"runtime/pprof"
	"strings"
	"syscall"
	"time"

	"github.com/mdp/qrterminal"

	"go.mau.fi/whatsmeow"
	"go.mau.fi/whatsmeow/appstate"
	"go.mau.fi/whatsmeow/store"
	"go.mau.fi/whatsmeow/store/sqlstore"
	"go.mau.fi/whatsmeow/types/events"
	waLog "go.mau.fi/whatsmeow/util/log"
)

// searchEnabled indicates whether vector/hybrid search is available.
// Set during NewMessageStore based on whether the embedder initialises successfully.
var searchEnabled bool

// package-level logger, initialized in main()
var logger waLog.Logger
var traceEnabled bool

// audioPipeline downloads and transcribes voice notes in the background.
// nil until Start() runs; its methods tolerate a nil receiver.
var audioPipeline *AudioPipeline

// tracef logs only when LOG_LEVEL=TRACE
func tracef(format string, args ...interface{}) {
	if traceEnabled {
		logger.Debugf("[TRACE] "+format, args...)
	}
}

func main() {
	// Set up logger
	logLevel := os.Getenv("LOG_LEVEL")
	if logLevel == "" {
		logLevel = "INFO"
	}
	if strings.ToUpper(logLevel) == "TRACE" {
		traceEnabled = true
		logLevel = "DEBUG"
	}
	logger = waLog.Stdout("Client", logLevel, true)

	reindex := flag.String("reindex", "", `rebuild the search index then exit; use "all" to reindex everything, or a partial JID/phone to reindex only matching chats`)
	cpuprofile := flag.String("cpuprofile", "", "write CPU profile to file (use with --reindex)")
	maxRows := flag.Int("max-rows", 0, "limit rows processed during --reindex (0 = unlimited, useful with --cpuprofile)")
	backfill := flag.Bool("transcribe-backfill", false, "transcribe all voice notes whose media is already on disk, then exit")
	backfillDays := flag.Int("transcribe-since-days", 0, "limit --transcribe-backfill to messages from the last N days (0 = all)")
	skipIndex := flag.Bool("skip-index", false, "with --transcribe-backfill: write transcripts to SQLite only, leaving the bleve index untouched so the bridge can keep running (requires a later --reindex all)")
	flag.Parse()

	if *backfill {
		open := NewMessageStore
		if *skipIndex {
			open = NewMessageStoreDBOnly
		}
		messageStore, err := open()
		if err != nil {
			logger.Errorf("Failed to initialise message store: %v", err)
			os.Exit(1)
		}
		if err := RunTranscribeBackfill(messageStore, *maxRows, *backfillDays); err != nil {
			logger.Errorf("Backfill failed: %v", err)
			messageStore.Close()
			os.Exit(1)
		}
		messageStore.Close()
		os.Exit(0)
	}

	if *reindex != "" {
		var profFile *os.File
		stopProfile := func() {
			if profFile != nil {
				pprof.StopCPUProfile()
				profFile.Close()
				profFile = nil
			}
		}

		if *cpuprofile != "" {
			f, err := os.Create(*cpuprofile)
			if err != nil {
				logger.Errorf("Could not create CPU profile: %v", err)
				os.Exit(1)
			}
			if err := pprof.StartCPUProfile(f); err != nil {
				logger.Errorf("Could not start CPU profile: %v", err)
				f.Close()
				os.Exit(1)
			}
			profFile = f

			// Flush profile on Ctrl+C
			sigs := make(chan os.Signal, 1)
			signal.Notify(sigs, syscall.SIGINT, syscall.SIGTERM)
			go func() {
				<-sigs
				logger.Infof("Interrupted — flushing CPU profile to %s", *cpuprofile)
				stopProfile()
				os.Exit(1)
			}()
		}

		chatFilter := *reindex
		if chatFilter == "all" {
			chatFilter = ""
			logger.Infof("--reindex all: deleting existing index at %s", indexPath)
			if err := deleteIndex(); err != nil {
				logger.Errorf("Failed to delete index: %v", err)
				stopProfile()
				os.Exit(1)
			}
		} else {
			logger.Infof("--reindex %q: reindexing matching chats only", chatFilter)
		}
		messageStore, err := NewMessageStore()
		if err != nil {
			logger.Errorf("Failed to initialise message store: %v", err)
			stopProfile()
			os.Exit(1)
		}
		if err := messageStore.ReIndexAllMessages(*maxRows, chatFilter); err != nil {
			logger.Errorf("Re-indexing failed: %v", err)
			messageStore.Close()
			stopProfile()
			os.Exit(1)
		}
		messageStore.Close()
		stopProfile()
		logger.Infof("Re-indexing complete, exiting.")
		os.Exit(0)
	}

	logger.Infof("Starting WhatsApp client...")

	// Update WhatsApp version to latest
	logger.Infof("Fetching latest WhatsApp version...")
	latestVersion, err := whatsmeow.GetLatestVersion(context.Background(), nil)
	if err != nil {
		logger.Warnf("Failed to fetch latest WhatsApp version, using default: %v", err)
	} else {
		store.SetWAVersion(*latestVersion)
		logger.Infof("Updated WhatsApp version to: %s", latestVersion.String())
	}

	// Create database connection for storing session data
	dbLog := waLog.Stdout("Database", logLevel, true)

	// Create directory for database if it doesn't exist
	if err := os.MkdirAll("store", 0755); err != nil {
		logger.Errorf("Failed to create store directory: %v", err)
		return
	}

	container, err := sqlstore.New(context.Background(), "sqlite3", "file:store/whatsapp.db?_foreign_keys=on", dbLog)
	if err != nil {
		logger.Errorf("Failed to connect to database: %v", err)
		return
	}

	// Get device store - This contains session information
	deviceStore, err := container.GetFirstDevice(context.Background())
	if err != nil {
		if err == sql.ErrNoRows {
			// No device exists, create one
			deviceStore = container.NewDevice()
			logger.Infof("Created new device")
		} else {
			logger.Errorf("Failed to get device: %v", err)
			return
		}
	}

	// Create client instance
	client := whatsmeow.NewClient(deviceStore, logger)
	if client == nil {
		logger.Errorf("Failed to create WhatsApp client")
		return
	} else {
		client.ManualHistorySyncDownload = true // Ensure we get all contacts before messages
	}

	// Initialize message store
	messageStore, err := NewMessageStore()
	if err != nil {
		logger.Errorf("Failed to initialize message store: %v", err)
		return
	}
	defer messageStore.Close()

	// Re-index existing messages
	go func() {
		err := messageStore.ReIndexAllMessages(0, "")
		if err != nil {
			logger.Errorf("Failed to re-index messages: %v", err)
		}
	}()

	broadcaster := NewMessageBroadcaster()

	registry, err := NewClientRegistry(messageStore.db)
	if err != nil {
		logger.Errorf("Failed to initialize client registry: %v", err)
		return
	}

	// Setup event handling for messages and history sync
	client.AddEventHandler(func(evt interface{}) {
		switch v := evt.(type) {
		case *events.Message:
			// Process regular messages and broadcast to WebSocket subscribers
			if bm := handleMessage(client, messageStore, v, logger); bm != nil {
				broadcaster.Broadcast(*bm)
			}

		case *events.Contact:
			tracef("Syncing Contacts!")

		case *events.HistorySync:
			// Process history sync events
			handleHistorySync(client, messageStore, v, logger)

		case *events.Connected:
			// STEP 1: Fetch contacts from critical_unblock_low app state
			logger.Infof("Step 1: Syncing contacts...")
			err := client.FetchAppState(context.Background(), appstate.WAPatchCriticalUnblockLow, true, false)
			if err != nil {
				logger.Infof("Error syncing contacts: %v", err)
				return
			}
			logger.Infof("Contacts synced successfully ✓")
			allContacts, err := client.Store.Contacts.GetAllContacts(context.Background())
			logger.Infof("Total contacts in store1: %d, error: %v", len(allContacts), err)

			client.ManualHistorySyncDownload = false // Now we can allow history sync to process messages

			// STEP 2: Fetch other critical app states
			logger.Infof("Step 2: Syncing critical block...")
			err = client.FetchAppState(context.Background(), appstate.WAPatchCriticalBlock, true, false)
			if err != nil {
				logger.Infof("Error syncing critical block: %v", err)
				return
			}
			logger.Infof("Critical block synced ✓")
			allContacts2, err := client.Store.Contacts.GetAllContacts(context.Background())
			logger.Infof("Total contacts in store2: %d, error: %v", len(allContacts2), err)

			// STEP 3: Sync other patches
			logger.Infof("Step 3: Syncing regular patches...")
			for _, patchName := range []appstate.WAPatchName{
				appstate.WAPatchRegularLow,
				appstate.WAPatchRegularHigh,
				appstate.WAPatchRegular,
			} {
				err := client.FetchAppState(context.Background(), patchName, true, false)
				if err != nil {
					logger.Infof("Error syncing %s: %v", patchName, err)
				} else {
					logger.Infof("%s synced ✓", patchName)
				}
			}

			// STEP 4: Now history sync is ready
			logger.Infof("Step 4: Ready for history sync...")

			allContacts3, err := client.Store.Contacts.GetAllContacts(context.Background())
			logger.Infof("Total contacts in store3: %d, error: %v", len(allContacts3), err)

			// History sync notifications will now be processed
		case *events.LoggedOut:
			logger.Warnf("Device logged out, please scan QR code to log in again")
		}
	})

	// Create channel to track connection success
	connected := make(chan bool, 1)

	// Connect to WhatsApp
	if client.Store.ID == nil {
		// No ID stored, this is a new client, need to pair with phone
		qrChan, _ := client.GetQRChannel(context.Background())
		err = client.Connect()
		if err != nil {
			logger.Errorf("Failed to connect: %v", err)
			return
		}

		// Print QR code for pairing with phone
		for evt := range qrChan {
			if evt.Event == "code" {
				logger.Infof("Scan this QR code with your WhatsApp app:")
				qrterminal.GenerateHalfBlock(evt.Code, qrterminal.L, os.Stdout)
			} else if evt.Event == "success" {
				connected <- true
				break
			}
		}

		// Wait for connection
		select {
		case <-connected:
			logger.Infof("Successfully connected and authenticated!")
		case <-time.After(3 * time.Minute):
			logger.Errorf("Timeout waiting for QR code scan")
			return
		}
	} else {
		// Already logged in, just connect
		err = client.Connect()
		if err != nil {
			logger.Errorf("Failed to connect: %v", err)
			return
		}
		connected <- true
	}

	// Wait a moment for connection to stabilize
	time.Sleep(2 * time.Second)

	if !client.IsConnected() {
		logger.Errorf("Failed to establish stable connection")
		return
	}

	logger.Infof("Connected to WhatsApp!")

	// Start voice-note download/transcription pipeline.
	audioPipeline = NewAudioPipeline(client, messageStore)
	audioPipeline.Start()
	defer audioPipeline.Stop()

	// Start REST API server
	startRESTServer(client, messageStore, broadcaster, registry, 8080)

	// Create a channel to keep the main goroutine alive
	exitChan := make(chan os.Signal, 1)
	signal.Notify(exitChan, syscall.SIGINT, syscall.SIGTERM)

	logger.Infof("REST server is running. Press Ctrl+C to disconnect and exit.")

	// Wait for termination signal
	<-exitChan

	logger.Infof("Disconnecting...")
	// Disconnect client
	client.Disconnect()
}
