#! /bin/sh

if ! [ -d experiment ]; then
    mkdir -p experiment/data
fi

cp $1/*.obj experiment/data
