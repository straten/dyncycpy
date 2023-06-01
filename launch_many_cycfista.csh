#!/usr/bin/bash -f

for bp in 0 1; do

  job0="b2"
  message0="bscrunch x 2"

  echo "both polns: $bp"

  if [ $bp = 1 ]; then
    perl -p -i -e 's|\.pb2|.b2|' cycfista.py
    job1="${job0}_bp"
    message1="${message0}, both polns"
  else
    perl -p -i -e 's|\.b2|.pb2|' cycfista.py
    job1=$job0
    message1=$message0
  fi

  # enforce causality by setting negative delays to zero
  for ec in 0 1; do

    echo "enforce causality: $ec"

    if [ $ec = 1 ]; then
      perl -p -i -e 's|CS.enforce_causality = False|CS.enforce_causality = True|' cycfista.py
      job2="${job1}_ec"
      message2="$message1, enforce causality"
    else
      perl -p -i -e 's|CS.enforce_causality = True|CS.enforce_causality = False|' cycfista.py
      job2=$job1
      message2=$message1
    fi

    # apply custom noise shrinkage threshold
    for st in 0 1; do

      echo "shrinkage threshold: $st"

      if [ $st = 1 ]; then
        perl -p -i -e 's|CS.noise_shrinkage_threshold = None|CS.noise_shrinkage_threshold = 1.0|' cycfista.py
        job3="${job2}_st"
        message3="$message2, noise shrinkage threshold"
      else
        perl -p -i -e 's|CS.noise_shrinkage_threshold = 1.0|CS.noise_shrinkage_threshold = None|' cycfista.py
        job3=$job2
        message3=$message2
      fi

      ./launch_cycfista.py $job3 "$message3"

    done
  done
done


