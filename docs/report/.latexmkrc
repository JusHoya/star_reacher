# latexmk configuration for the scientific report build (FR-30).
#
# Mirrors docs/mathlib/.latexmkrc: the document version is captured from
# `git describe --always` at build time so both publication artifacts trace
# to the same source revision. version.tex is git-ignored build litter; the
# document falls back to "unknown" when it is absent.

$pdf_mode = 1;        # always produce PDF via pdflatex
$bibtex_use = 2;      # run biber as needed and clean generated .bbl on -C

my $version = `git describe --always --dirty`;
if ($? != 0 or !defined($version)) { $version = ''; }
chomp $version;
# Map underscores to hyphens and drop characters outside a conservative safe
# set so a git description can never inject TeX-active characters.
$version =~ s/_/-/g;
$version =~ s/[^A-Za-z0-9.\-+\/]//g;
$version = 'unknown' if $version eq '';

open(my $fh, '>', 'version.tex') or die "cannot write version.tex: $!";
print $fh "\\renewcommand{\\starversion}{$version}\n";
close($fh);
