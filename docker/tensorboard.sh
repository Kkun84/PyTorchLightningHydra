#!/bin/bash
docker exec -itd PyTorchLightningHydra tensorboard --logdir=. --host=0.0.0.0 --port=${@-6006}
