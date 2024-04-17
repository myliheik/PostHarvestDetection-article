module load geoconda

# no need to edit 

echo Generate command list for set Cloudless

if test -f komennotCloudless.txt; then
    rm komennotCloudless.txt
fi

touch komennotCloudless.txt


echo Choosing tiles...

# Each data set (shp) are spread on different set of tiles. We choose only relevant tiles.

#ls -1 $shppath | cut -f4 -d\_ | cut -f1 -d\. | sort | uniq > tiles.txt

#grep -f 6-tilesAOI.txt safepaths.txt > safepaths2.txt






#FILES="confs/configCloudless*"
#FILES="confs/configCloudless.config202*insitu" # 2020-2022 kuvat hävinny
FILES="confs/configCloudless.config2023insitu"

for setti in $FILES
do 
. "$setti"

echo The shapefile path is $shppath
echo The project path is $projectpath
#break

rivi_lkm=$(cat $safepath | wc -l)

echo Write commands to a file...

for ID in $( seq 1 $rivi_lkm)
do
   name=$(sed -n ${ID}p $safepath)
   echo "python /projappl/project_2002224/vegcover2023/python/03-arrayextractor-S2.py -f $name -shp $shppath -p $projectpath -jn ${ID} -id $idname -r 10 -t \$LOCAL_SCRATCH" >> komennotCloudless.txt
done




#polku=/scratch/project_2002224/vegcover2023/cloudless/results-2020
    if [[ ! -d $projectpath ]]
    then
        echo "Creating directory ${projectpath}"
        mkdir -p $projectpath
    fi

done

echo Muista module load geoconda !
echo Run sbatch_commandlist -commands komennotCloudless.txt -t 00:30:00 -mem 9000 --tmp 2000

echo sbatch_commandlist -commands komennotCloudless.txt -t 00:20:00 -mem 1500 --tmp 200 # pitäis riittää, mut laitetaan silti 2000
