#!/bin/sh

./halite --replay-directory replays/ -vvv --width 32 --height 32 "python3.6 MyBot.py" "python3.6 dummy.py"
./halite --replay-directory replays/ -vvv --width 32 --height 32 "python3.6 MyBot.py" "python3.6 dummy.py"
./halite --replay-directory replays/ -vvv --width 32 --height 32 "python3.6 MyBot.py" "python3.6 dummy.py"
./halite --replay-directory replays/ -vvv --width 32 --height 32 "python3.6 MyBot.py" "python3.6 dummy.py"
./halite --replay-directory replays/ -vvv --width 32 --height 32 "python3.6 MyBot.py" "python3.6 dummy.py"

python3.6 policy_update.py
