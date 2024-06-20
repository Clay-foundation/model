## Download data
The data comes as multifile zip, it can be downloaded from the
[BioMassters](https://huggingface.co/datasets/nascetti-a/BioMassters/)
huggingface repository. Grab a coffee, this is about 250GB in size.

The next step is to unzip training data. The data comes in a multi-file
zip archive. So it needs to be unzipped using a library that can handle
the format. 7z works quite well in this case. Grabb another coffee, this
will take a while.

```bash
sudo apt install p7zip-full
```

### Extract train feature

```bash
7z e -o/home/tam/Desktop/biomasters/train_features/ /datadisk/biomasters/raw/train_features.zip
```

Should look something like this

```
7-Zip [64] 16.02 : Copyright (c) 1999-2016 Igor Pavlov : 2016-05-21
p7zip Version 16.02 (locale=en_US.UTF-8,Utf16=on,HugeFiles=on,64 bits,16 CPUs Intel(R) Core(TM) i7-10875H CPU @ 2.30GHz (A0652),ASM,AES-NI)

Scanning the drive for archives:
1 file, 10247884383 bytes (9774 MiB)

Extracting archive: /datadisk/biomasters/raw/train_features.zip
--
Path = /datadisk/biomasters/raw/train_features.zip
Type = zip
Physical Size = 10247884383
Embedded Stub Size = 4
64-bit = +
Total Physical Size = 149834321503
Multivolume = +
Volume Index = 13
Volumes = 14

Everything is Ok

Folders: 1
Files: 189078
Size:       231859243932
Compressed: 149834321503
```

### Extract train AGBM

```bash
7z e -o/home/tam/Desktop/biomasters/train_agbm/ /datadisk/biomasters/raw/train_agbm.zip
```

Should look something like this

```
7-Zip [64] 16.02 : Copyright (c) 1999-2016 Igor Pavlov : 2016-05-21
p7zip Version 16.02 (locale=en_US.UTF-8,Utf16=on,HugeFiles=on,64 bits,16 CPUs Intel(R) Core(TM) i7-10875H CPU @ 2.30GHz (A0652),ASM,AES-NI)

Scanning the drive for archives:
1 file, 575973495 bytes (550 MiB)

Extracting archive: /datadisk/biomasters/raw/train_agbm.zip
--
Path = /datadisk/biomasters/raw/train_agbm.zip
Type = zip
Physical Size = 575973495

Everything is Ok

Folders: 1
Files: 8689
Size:       2280706098
Compressed: 575973495
```

### Extract test features

```bash
7z e -o/home/tam/Desktop/biomasters/test_features/ /datadisk/biomasters/raw/test_features_splits.zip
```

Should look something like this

```
7-Zip [64] 16.02 : Copyright (c) 1999-2016 Igor Pavlov : 2016-05-21
p7zip Version 16.02 (locale=en_US.UTF-8,Utf16=on,HugeFiles=on,64 bits,16 CPUs Intel(R) Core(TM) i7-10875H CPU @ 2.30GHz (A0652),ASM,AES-NI)

Scanning the drive for archives:
1 file, 6912625480 bytes (6593 MiB)

Extracting archive: /datadisk/biomasters/raw/test_features_splits.zip
--
Path = /datadisk/biomasters/raw/test_features_splits.zip
Type = zip
Physical Size = 6912625480
Embedded Stub Size = 4
64-bit = +
Total Physical Size = 49862298440
Multivolume = +
Volume Index = 4
Volumes = 5

Everything is Ok

Folders: 1
Files: 63348
Size:       78334396224
Compressed: 49862298440
```

### Extract test AGBM

```bash
7z e -o/home/tam/Desktop/biomasters/test_agbm/ /datadisk/biomasters/raw/test_agbm.tar
```

Should look something like this

```
7-Zip [64] 16.02 : Copyright (c) 1999-2016 Igor Pavlov : 2016-05-21
p7zip Version 16.02 (locale=en_US.UTF-8,Utf16=on,HugeFiles=on,64 bits,16 CPUs Intel(R) Core(TM) i7-10875H CPU @ 2.30GHz (A0652),ASM,AES-NI)

Scanning the drive for archives:
1 file, 729766400 bytes (696 MiB)

Extracting archive: /datadisk/biomasters/raw/test_agbm.tar
--
Path = /datadisk/biomasters/raw/test_agbm.tar
Type = tar
Physical Size = 729766400
Headers Size = 1421312
Code Page = UTF-8

Everything is Ok

Folders: 1
Files: 2773
Size:       727862586
Compressed: 729766400
```

## Prepare data

This will take the average of all timesteps available for each tile.
The time steps for Sentinel-2 are not complete, not all months are
provided for all tiles. In addtion, the Clay model does not take time
series as input. So aggregating the time element is simplifying but
ok for the purpose of this example.

### Prepare training features

```bash
python finetune/regression/preprocess_data.py \
  --features=/home/tam/Desktop/biomasters/train_features/ \
  --cubes=/home/tam/Desktop/biomasters/train_cubes/ \
  --processes=12 \
  --sample=1
```

### Prepare test features

```bash
python finetune/regression/preprocess_data.py \
  --features=/home/tam/Desktop/biomasters/test_features/ \
  --cubes=/home/tam/Desktop/biomasters/test_cubes/ \
  --processes=12 \
  --sample=1
```
