#!/usr/bin/perl

use strict;
use Data::Dumper;

use Algorithm::Diff::XS qw(traverse_balanced);

$Data::Dumper::Indent = 0;

while(<STDIN>) {
  chomp;
  my ($src, $trg, $aln) = split(/\t/, $_);
  
  #print $src, "\n";
  #print $trg, "\n";
  
  my @src = split(/\s/, $src);
  my @trg = split(/\s/, $trg);
  
  #print scalar @src, " ", scalar @trg, "\n";

  push(@src, "</s>");
  push(@trg, "</s>");
  
  my @seq; 
  traverse_balanced(
      \@trg, \@src,
      {   MATCH => sub { push(@seq, "<step>", $trg[$_[0]]); },
          DISCARD_A => sub { push(@seq, $trg[$_[0]]); },
          DISCARD_B => sub { push(@seq, "<step>"); },
          CHANGE    => sub { push(@seq, "<step>", $trg[$_[0]]); },
      }
  );
  
  shift(@seq);
  pop(@seq);
  print join(" ", @seq), "\n";
  
  #my @aln = sort { $a->[1] <=> $b->[1] or $a->[0] <=> $b->[0] }
  #  map {[map { $_ + 0 } split(/-/, $_)]} split(/\s/, $aln);
  #
  #push(@aln, [scalar @src, scalar @trg]);
  #
  #print Dumper(\@aln), "\n";
  #
  #my @t;
  #foreach my $p (@aln) {
  #  $t[$p->[1]] = $p->[0] if not defined $t[$p->[1]];
  #}
  #
  #print Dumper(\@t), "\n";
  #
  #my $c = 0;
  #foreach my $t (@t) {
  #  if($t > $c) {
  #    $c = $t;
  #  }
  #  else {
  #    $t = $c;
  #  }
  #}
  
  #print Dumper(\@t), "\n";
}