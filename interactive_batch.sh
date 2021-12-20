#!/bin/bash
srun --pty -c 2 -t 2:00:00 --mem=8G -G 1 -p gpu_devel bash
