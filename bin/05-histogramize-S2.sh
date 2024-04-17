#!/bin/bash -l

module load geoconda

# no need to edit

FILENAME=cloudless-histo-komento.txt

echo Generate command list for set Cloudless

if test -f $FILENAME; then
    rm $FILENAME
fi

touch $FILENAME

bands=("B02" "B03" "B04" "B05" "B06" "B07" "B08" "B8A" "B11" "B12")

#FILES="confs/configCloudless.config202*"
FILES="confs/configCloudless.config2023insitu"

for setti in $FILES
do
. "$setti"

	echo The project path is $projectpath

        INPUT=$projectpath
        OUTPUT=$histopath

        echo For each band...
                for bandi in ${bands[@]}
                do
                        . /projappl/project_2002224/vegcover2023/bin/confs/$bandi.config
                        echo "python /projappl/project_2002224/vegcover2023/python/05-histogramize-shadow.py -i $INPUT -o $OUTPUT -b $bandi -n 32 -l $minimi -u $maksimi" >> $FILENAME
                done

done

echo Running cloudless histo commands...
echo sbatch_commandlist -commands cloudless-histo-komento.txt

