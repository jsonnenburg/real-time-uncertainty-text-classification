#!/bin/bash

remote_user="sonnenbj"
remote_host="gruenau1.informatik.hu-berlin.de"
remote_dir="/vol/fob-vol1/nebenf23/sonnenbj/real-time-uncertainty-text-classification/out/bert_student"
local_dir="/Users/johann/Documents/Uni/real-time-uncertainty-text-classification/out/bert_student"

# Use rsync to copy files maintaining the directory structure
rsync -avz --include "*/" --include "results/results.json" --exclude "*" -e ssh "$remote_user@$remote_host:$remote_dir" "$local_dir"
