#!/bin/bash

# Copy over the saved model file for p01 scan 1
cp /rds/projects/c/chechlmy-chbh-mricn/axs2210/feat/1/p01_s1.feat/design.fsf p01_s1.fsf

# Create model file for p02 scan1
cp p01_s1.fsf p02_s1.fsf
perl -i -p -e s/p01/p02/ p02_s1.fsf
  /recon/p02/fmri1
  /feat/1/p02_s1
  /recon/p02/T1_brain

# Create model files for p03-p15 scan 1
for p in p03 p04 p05 p06 p07 p08 p09 p10 p11 p12 p13 p14 p15
do
   cp p01_s1.fsf ${p}_s1.fsf
   perl -i -p -e s/p01/${p}/ ${p}_s1.fsf
   perl -i -p -e s/93/94/    ${p}_s1.fsf
done

# Create model files for p03-p15 scan 2
for p in p03 p04 p05 p06 p07 p08 p09 p10 p11 p12 p13 p14 p15
do
   cp ${p}_s1.fsf ${p}_s2.fsf
   perl -i -p -e 's/_s1/_s2/'     ${p}_s2.fsf
   perl -i -p -e 's/fmri1/fmri2/' ${p}_s2.fsf
done

# Create model files for p05 scan 3
cp p05_s1.fsf p05_s3.fsf
perl -i -p -e 's/_s1/_s3/'     p05_s3.fsf
perl -i -p -e 's/fmri1/fmri3/' p05_s3.fsf
