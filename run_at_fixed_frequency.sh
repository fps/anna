sudo cpupower frequency-set -u $1

shift

$@ 

sudo cpupower frequency-set -u 1THz
