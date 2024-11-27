#!/bin/bash

# Define the directory to search, replace with your specific directory path
directory="/home/deepl/OCR_LABEL/PaddleOCR/最终的图片2_1024size/"

# Use find command to traverse all subfolders
# Find files starting with 'new_model' with extensions 'jpg' or 'png'
find "$directory" -type f \( -name "new_model*.jpg" -o -name "new_model*.png" \) -exec rm {} \;

# Add another find command to delete all txt files
find "$directory" -type f -name "*.txt" -exec rm {} \;

echo "All files starting with 'new_model' and ending with '.jpg' or '.png' have been deleted."
echo "All '.txt' files have been deleted."

