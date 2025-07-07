curl https://github.com/i-machine-think/am-i-compositional/archive/refs/heads/master.zip -L -J -O
unzip am-i-compositional-master.zip
rm -r am-i-compositional-master.zip
mv am-i-compositional-master/data/pcfgset PCFGS
rm -r am-i-compositional-master
