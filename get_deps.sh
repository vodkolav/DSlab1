outname=$1.yml
conda-minify --name $1  --how=major -f $outname

# sort deps by name
# head -6 tmp > $outname
#TL=$(wc -l tmp)
# tail - tmp | sort >> $outname
# rm tmp

#outname=$1_reqs.txt
#reqs=$(conda-minify --name $1  --how=minor)

#head -6 conda_req.txt > conda_req_sorted.txt 
#tail -69 conda_req.txt | sort >> conda_req_sorted.txt

