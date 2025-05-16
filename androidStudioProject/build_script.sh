#!/bin/bash
echo "Running build with verbose output..."
./gradlew clean
./gradlew --debug compileDebugKotlin > debug_log.txt 2>&1
echo "Build completed. Searching for errors..."
grep -n "error:" debug_log.txt -A 5 -B 5 > errors.txt
echo "Errors found:"
cat errors.txt