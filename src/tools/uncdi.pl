#!/usr/bin/perl

use strict;

open(SRC, "<$ARGV[0]") or die;

while(<STDIN>) {
  chomp;
  my $cdi = $_;
  chomp(my $src = <SRC>);
  
  #print $src, "\n";
  #print $cdi, "\n";
  
  my @src = split(/\s/, $src);
  my @cdi = split(/\s/, $cdi);
  
  push(@src, "</s>");
  
  my $cur = 0;
  
  my @out;
  foreach my $act (@cdi) {
    if($act eq "<c>") {
      push(@out, $src[$cur]);
      $cur++;
    }
    elsif($act eq "<d>") {
      $cur++;
    }
    else {
      push(@out, $act);
    }
  }
  
  pop(@out);
  
  print join(" ", @out), "\n";  
}