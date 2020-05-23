#!/bin/bash

PID=9074
while [ -e /proc/$PID  ]
do
    echo "Process: $PID is still running"
    sleep 10
done

echo "start jobs for book corpus"

nohup bash job.sh book > book_all.log 2>&1 &
