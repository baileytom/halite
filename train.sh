#!/bin/sh

for i in $(seq 1 $2)
do
  echo batch $i  
  for j in $(seq 1 $1)
  do
    echo episode $j  
    ./halite --replay-directory replays/ -vvv --width 32 --height 32 "python3.6 MyBot.py ${j}" "python3.6 dummy.py"
  done
  python3.6 update.py
done
