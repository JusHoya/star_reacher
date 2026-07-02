# latexmk configuration for the math library build (FR-29).
#
# The document version is captured from `git describe --always` at build time
# rather than hard-coded in the LaTeX source, so every built PDF names the
# exact source revision it was produced from without manual bookkeeping.
# The generated version.tex is build litter (git-ignored); the document
# falls back to "unknown" when it is absent.

$pdf_mode = 1;        # always produce PDF via pdflatex
$bibtex_use = 2;      # run biber as needed and clean generated .bbl on -C

my $version = `git describe --always --dirty`;
if ($? != 0 or !defined($version)) { $version = ''; }
chomp $version;
# Underscores and other TeX-active characters in a git description would
# break typesetting in \texttt; map underscores to hyphens and drop anything
# else outside a conservative safe set.
$version =~ s/_/-/g;
$version =~ s/[^A-Za-z0-9.\-+\/]//g;
$version = 'unknown' if $version eq '';

open(my $fh, '>', 'version.tex') or die "cannot write version.tex: $!";
print $fh "\\renewcommand{\\starversion}{$version}\n";
close($fh);
