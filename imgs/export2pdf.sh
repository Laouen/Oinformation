# !/bin/bash
# make script executable: chmod +x export2pdf.sh

# export all .svg in folder to .pdf
for i in *.svg; do inkscape --file="$i" --without-gui --export-pdf="${i%.svg}.pdf"; done