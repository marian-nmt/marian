#!/usr/bin/perl

use strict;
use Data::Dumper;

# es ist nicht möglich , geben Sie den Text direkt in die Zeichnung .
# es ist nicht möglich , Text direkt in die Zeichnung einzugeben .
# <c> <c> <c> <c> <c> <d> <d> <d> <c> <c> <c> <c> <c> einzugeben <c>


use Algorithm::Diff::XS qw(traverse_balanced);

$Data::Dumper::Indent = 0;

while(<STDIN>) {
  chomp;
  my ($src, $trg, $aln) = split(/\t/, $_);
  
  #print $src, "\n";
  #print $trg, "\n";
  
  my @src = split(/\s/, $src);
  my @trg = split(/\s/, $trg);

  my @seq; 
  traverse_balanced(
      \@trg, \@src,
      {   MATCH => sub { push(@seq, "<c>"); },
          DISCARD_A => sub { push(@seq, $trg[$_[0]]); },
          DISCARD_B => sub { push(@seq, "<d>"); },
          CHANGE    => sub { push(@seq, "<d>", $trg[$_[0]]); },
      }
  );
  
  print join(" ", @seq), "\n";
}