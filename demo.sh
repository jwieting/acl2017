#load and evaluate pre-trained GRAN model
if [ "$1" == "load-gran" ]; then
    sh train.sh -model gran -outfile gran -wordfile ../data/paragram_sl999_small.txt -loadmodel ../data/gran.pickle

#train sentence similarity models
elif [ "$1" == "gran" ]; then
    sh train.sh -model gran -wordfile ../data/paragram_sl999_small.txt -gran_type 1 -scramble 0.5 -dropout 0.4
elif [ "$1" == "lstm" ]; then
    sh train.sh -model lstm -wordfile ../data/paragram_sl999_small.txt -LW 1e-05 -scramble 0.75 -word_dropout 0.2
elif [ "$1" == "lstmavg" ]; then
    sh train.sh -model lstmavg -wordfile ../data/paragram_sl999_small.txt -LW 1e-05 -scramble 0.5 -dropout 0.4
elif [ "$1" == "wordaverage" ]; then
    sh train.sh -model wordaverage -wordfile ../data/paragram_sl999_small.txt -margin 0.6
else
    echo "$1 not a valid option."
fi
