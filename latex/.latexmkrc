# Build with: cd latex && latexmk
$pdf_mode = 1;           # pdflatex (same engine Overleaf uses by default)
$out_dir = 'build';      # keep aux files out of the source tree
@default_files = ('main.tex');
$pdflatex = 'pdflatex -interaction=nonstopmode -synctex=1 -file-line-error %O %S';
