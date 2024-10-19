#!/bin/bash

git add *.dvc
git add .gitignore

git commit -m "Retraining complete"

git push
dvc push