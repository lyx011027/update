

#!/bin/bash

if [ ! $# == 3 ]; then

echo "Usage: $0 CPUs DIMMs mem_err_events"

exit

fi

CPUs="$1"
DIMMs="$2"
mem_err_events="$3"
# python3 split.py "$CPUs" "$DIMMs" "$mem_err_events" 
python3 count.py "$CPUs" "$DIMMs" "$mem_err_events" 
# python3 sample_multipro.py
# python3 rf_multipro.py

