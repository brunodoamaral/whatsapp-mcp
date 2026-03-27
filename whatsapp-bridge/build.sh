#!/bin/bash

CGO_LDFLAGS="-L/usr/local/lib" CGO_CFLAGS="-I/usr/local/include" go build -tags "vectors full" -o whatsapp-bridge-bin .
