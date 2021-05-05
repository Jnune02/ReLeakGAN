#!/bin/sh

# Autostart ReLeakGan Demo
alias python="/usr/bin/python3"
BASE_DIR=${PWD##*/}

CWD=`pwd`

if [ $BASE_DIR != "ReLeakGan" ]; then
    echo "autostart.sh must be run from base directory: $BASE_DIR !!!"
    exit 1
fi

if [ -a ./sout.log ]; then
    rm ./sout.log
fi

cd ./sourceTexts/
python ./package_corpus.py | tee -a $CWD/sout.log
cd $CWD
    
cp ./sourceTexts/data/* ./data/
cp ./data/train_corpus_padded.npy ./data/train_corpus.npy


# Supress Standard Error

exec 3>&2
exec 2> /dev/null

# clean stale '~' files
rm ./*~
rm ./sourceTexts/*~
rm ./sourceTexts/data/*~
rm ./data/*~

# Restore Standard Error
exec 2>&3

echo "STDOUT messages saved in sout.log" | tee -a $CWD/sout.log

# start main application entry point
echo "Starting Main Application Entry Point" | tee -a $CWD/sout.log
python ./main.py | tee -a $CWD/sout.log

# start generating trained output
echo "Finished training. Extracting Results" | tee -a $CWD/sout.log
python ./extract_checkpoints.py | tee -a $CWD/sout.log

# start evaluation using automated metrics
echo "Finished extraction. Beginning Metric Analysis." | tee -a $CWM/sout.log
cp -R ./checkpoints/data/* ./metrics/data/
python ./metric_analysis.py | tee -a $CWD/sout.log

# process complete.
echo "Process complete. Exit."
exit 0

