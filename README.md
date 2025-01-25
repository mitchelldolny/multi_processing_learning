# multi_processing_learning
This repository is simply me learning how powerful multiprocessing is. Back when I did my plant trait prediction project I had numerous problems loading in the 43k images that the program required for training/validation/testing purposes. Until today I've never utilized multiprocessing and was curious on how powerful it could be. So I tested the following transformation against the dataset originally used and the multiprocessed one.

Here are the results:
Multi_Processed:
100%|██████████████████████████████████████████████████████████████████| 43363/43363 [00:00<00:00, 506392.93it/s]
Multiprocessing image loading time: 28.66 seconds

Non_Multi_Processed:
80%|███████████████████████████████████████████████████████▉              | 34681/43363 [02:01<17:14,  8.39it/s]
it crashed my computer and ran out of memory, thus could not load in the entire dataset needed for training.

Computer Specs for whoever is interested:
Apple M3 Pro chip, with 36GB of unified memory

Hope this helps!

Futher note:
- Attempted just threading which took way longer
- Combined both threading and multiprocessing and it took ~32 vs ~26 seconds
