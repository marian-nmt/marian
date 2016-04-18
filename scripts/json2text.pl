#!/usr/bin/env perl
use strict;
use JSON;
use Data::Dumper;

my $json = "";
while(<STDIN>) {
  $json .= $_;
}

my $d = from_json($json);
my @v;
foreach my $k (keys %$d) {
  $v[$d->{$k}] = $k;
}
print "$_\n" foreach(@v);
