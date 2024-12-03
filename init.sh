#!/bin/bash -e

if [ -d SaProt ]; then
    echo "It seems that you have already run this installation script before. You only need to run this script once."
    echo "However, if you want a fresh installation, please delete the SaProt directory and run this script again."
    exit 1
fi

# Download SaProt
git clone https://github.com/westlake-repl/SaProt
cd SaProt
git reset --hard 9f83e6c71ffd142e0244caf52114221ee352b594
rm -rf .git
cd bin

# Fix import problem in SaProt
sed -i '6s/.*/from ..utils.lr_scheduler import Esm2LRScheduler/' ../model/abstract_model.py

# Download Foldseek
if [ "$1" = "avx2" ]; then
    wget https://mmseqs.com/foldseek/foldseek-linux-avx2.tar.gz
elif [ "$1" = "sse2" ]; then
    wget https://mmseqs.com/foldseek/foldseek-linux-sse2.tar.gz
elif [ "$1" = "arm64" ]; then
    wget https://mmseqs.com/foldseek/foldseek-linux-arm64.tar.gz
elif [ "$1" = "osx" ]; then
    wget https://mmseqs.com/foldseek/foldseek-osx-universal.tar.gz
else
    echo "Unrecognized argument!"
    echo "Valid arguments are \"avx2\" (for Linux AVX2 build), \"sse2\" (for Linux SSE2 build), \"arm64\" (for Linux ARM64 build), and \"osx\" (for macOS)." 
    cd ../../
    rm -rf SaProt
    exit 1
fi

# Unzip the Foldseek tarball, and then delete the tarball
foldseek=$(awk -F"/" '{print $NF}' <<< "$_") 
tar -xvzf "$foldseek"
rm -f "$foldseek"

# Place the Foldseek binary in the bin folder of SaProt
mv foldseek/bin/foldseek foldseek/bin/foldseek-bin
mv foldseek/bin/foldseek-bin .
rm -r foldseek
mv foldseek-bin foldseek

# Download SaProt checkpoint (SaProt_650M_AF2)
cd ../
git lfs install
echo "============================================================================================================"
echo "NOTICE:"
echo "This git clone step may take a few minutes since SaProt (around 5 GB) is being downloaded from Hugging Face."
echo "============================================================================================================"
git clone https://huggingface.co/westlake-repl/SaProt_650M_AF2
