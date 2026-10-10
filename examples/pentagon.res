# Max-Cut on a 5-cycle plus one chord. Best cut = 5.
var a b c d e
maxcut a b 1
maxcut b c 1
maxcut c d 1
maxcut d e 1
maxcut e a 1
maxcut a c 1
run trials=16
