#!/bin/bash

git add *.dvc
git add .gitignore
git add dvc.lock

git commit -m "Retraining complete"

git push
dvc push