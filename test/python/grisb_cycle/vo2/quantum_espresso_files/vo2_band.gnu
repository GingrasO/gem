set style data dots
set nokey
set xrange [0: 5.84307]
set yrange [  2.23144 : 17.57901]
set arrow from  0.69550,   2.23144 to  0.69550,  17.57901 nohead
set arrow from  1.39101,   2.23144 to  1.39101,  17.57901 nohead
set arrow from  2.37460,   2.23144 to  2.37460,  17.57901 nohead
set arrow from  3.46847,   2.23144 to  3.46847,  17.57901 nohead
set arrow from  4.16397,   2.23144 to  4.16397,  17.57901 nohead
set arrow from  4.85948,   2.23144 to  4.85948,  17.57901 nohead
set xtics ("G"  0.00000,"X"  0.69550,"M"  1.39101,"G"  2.37460,"Z"  3.46847,"R"  4.16397,"A"  4.85948,"Z"  5.84307)
 plot "vo2_band.dat"
