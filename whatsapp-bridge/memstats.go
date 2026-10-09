package main

/*
#include <malloc.h>
*/
import "C"

import (
	"net/http"
	"net/http/pprof"
	"os"
	"runtime"
	"time"
)

// init pins glibc malloc's large-allocation behaviour before any native
// library (ONNX Runtime, FAISS) allocates. By default glibc raises its mmap
// threshold every time a large mmapped chunk is freed (up to 32 MB), after which
// ORT's per-inference activation buffers (several MB each) are carved from the
// per-thread heap arenas instead, and fragment there. cgo calls hop between OS
// threads, so up to 8×cores arenas each keep a high-water mark. Measured with
// TestEmbedderMemory over 300 mixed batches: the default swung 550–935 MB,
// while a fixed 128 KiB threshold stayed flat at ~260 MB at the same speed.
// Setting the threshold explicitly also disables the dynamic adjustment.
// M_ARENA_MAX=2 bounds the arena count as a second guard.
func init() {
	C.mallopt(C.M_MMAP_THRESHOLD, 128*1024)
	C.mallopt(C.M_ARENA_MAX, 2)
}

// memReportInterval is how often the bridge logs its memory split. Most of the
// process's memory is C heap (ONNX Runtime, FAISS, SQLite) that Go's profiler
// can't see, so the useful number is total minus Go: if `c` grows while `go_sys`
// stays flat, the leak is native; otherwise `go tool pprof` will find it.
const memReportInterval = 10 * time.Minute

// startMemReporter logs "Memory: anon=… go_sys=… go_heap=… c≈…" (MB) every
// memReportInterval, so memory growth can be read straight from the journal.
func startMemReporter() {
	go func() {
		for {
			logMemSplit()
			time.Sleep(memReportInterval)
		}
	}()
}

func logMemSplit() {
	var ms runtime.MemStats
	runtime.ReadMemStats(&ms)
	anon := procMemKB() / 1024
	goSys := int64(ms.Sys-ms.HeapReleased) >> 20
	logger.Infof("Memory: anon=%dMB go_sys=%dMB go_heap=%dMB c≈%dMB goroutines=%d",
		anon, goSys, int64(ms.HeapAlloc)>>20, anon-goSys, runtime.NumGoroutine())
}

// startPprofServer serves net/http/pprof on PPROF_ADDR (e.g. 127.0.0.1:6060)
// when set. It gets its own listener rather than a route on the REST router,
// which binds all interfaces. Heap profiles cover Go allocations only.
func startPprofServer() {
	addr := os.Getenv("PPROF_ADDR")
	if addr == "" {
		return
	}
	mux := http.NewServeMux()
	mux.HandleFunc("/debug/pprof/", pprof.Index)
	mux.HandleFunc("/debug/pprof/cmdline", pprof.Cmdline)
	mux.HandleFunc("/debug/pprof/profile", pprof.Profile)
	mux.HandleFunc("/debug/pprof/symbol", pprof.Symbol)
	mux.HandleFunc("/debug/pprof/trace", pprof.Trace)
	go func() {
		logger.Infof("pprof listening on http://%s/debug/pprof/", addr)
		if err := http.ListenAndServe(addr, mux); err != nil {
			logger.Warnf("pprof server stopped: %v", err)
		}
	}()
}
