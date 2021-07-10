#!/bin/bash
######################################################a#
# The following is a description of the script content
# Name: CZA
# Path: /home/user26/design/bin
# Update: 2020/10/12 22:51
# Usage: Get the difference of the top 1 and top2.
######################################################a#
Accuracy=0
for((g=1;g<=10;g++))
do
for((j=1+10*g-10;j<=10+10*g-10;j++))
do
	n=0
	line=4
	r=3030
	first[1]=`cat /home/user68/DNN/ReRAM Array/dnn$j.mt0|sed -n "${line}p"|awk '{print $1}'`
	first[2]=`cat /home/user68/DNN/ReRAM Array/dnn$j.mt0|sed -n "${line}p"|awk '{print $2}'`
	first[3]=`cat /home/user68/DNN/ReRAM Array/dnn$j.mt0|sed -n "${line}p"|awk '{print $3}'`
	first[4]=`cat /home/user68/DNN/ReRAM Array/dnn$j.mt0|sed -n "${line}p"|awk '{print $4}'`
	first[5]=`cat /home/user68/DNN/ReRAM Array/dnn$j.mt0|sed -n "${line}p"|awk '{print $5}'`
	first[6]=`cat /home/user68/DNN/ReRAM Array/dnn$j.mt0|sed -n "${line}p"|awk '{print $6}'`
	first[7]=`cat /home/user68/DNN/ReRAM Array/dnn$j.mt0|sed -n "${line}p"|awk '{print $7}'`
	first[8]=`cat /home/user68/DNN/ReRAM Array/dnn$j.mt0|sed -n "${line}p"|awk '{print $8}'`
	first[9]=`cat /home/user68/DNN/ReRAM Array/dnn$j.mt0|sed -n "${line}p"|awk '{print $9}'`
	first[10]=`cat /home/user68/DNN/ReRAM Array/dnn$j.mt0|sed -n "${line}p"|awk '{print $10}'`
	#for((i=1;i<=10;i++))
	#do
		#echo "${first[$i]}"
	#done

	for((i=1;i<=10;i++))
	do
	  # 删掉最后一个 e 及其右边的字符串
		num[$i]=${first[$i]%e*}
	done

	for((i=1;i<=10;i++))
	do
		# 删掉最后一个 . 及其右边的字符串
		num2[$i]=${num[$i]%.*}
	done

#q=`echo $y | cut -d":" -f4`
	for((i=1;i<=10;i++))
	do
		# 删掉第一个.及其左边的字符串
		num3[$i]=${num[$i]#*.}
	done

	for((i=1;i<=10;i++))
	do
	  # 小数点后一位
		shu1[$i]=`echo ${num3[$i]}|cut -c 1`
		# 小数点后二位
		shu2[$i]=`echo ${num3[$i]}|cut -c 2`
		# 小数点后三位
		shu3[$i]=`echo ${num3[$i]}|cut -c 3`
		# 小数点后四位
		shu4[$i]=`echo ${num3[$i]}|cut -c 4`
		#echo "${shu1[$i]}"
	done

	for((i=1;i<=10;i++))
	do
	  # 将小数点以前的乘以10000
		num4[$i]=$((${num2[$i]}*10000))
	done

	for((i=1;i<=10;i++))
	do
    # 其实就是将整体乘以10000倍
		final[$i]=$((${num4[$i]}+${shu1[i]}*1000+${shu2[i]}*100+${shu3[i]}*10+${shu4[i]}))
		#echo "${final[$i]}"
	done

	for((i=1;i<=g-1;i++))
	do
		a=${final[g]}
		b=${final[$i]}
		diff[$i]=$(($a-$b))
		#echo ${diff[$i]}
	done

	for((i=g+1;i<=10;i++))
	do
		a=${final[g]}
		b=${final[$i]}
		diff[$i-1]=$(($a-$b))
		#echo ${diff[$i-1]}
	done

	#for((i=1;i<=9;i++))
	#do
	#echo "${diff[$i]}"
	#done

	ref=${diff[1]}
	#echo ${ref}
	result=2
	wi=-1

	for((i=1;i<=9;i++))
	do
		if [[  $ref -gt ${diff[i]} ]] ; then
			result=$result
			#echo 1
			ref=${diff[$i]}
		else
			result=$i
			ref=${ref}
		fi
	done

	echo "${ref}"

	if [[ $ref -gt $n ]] ; then
			Accuracy=$(($Accuracy+1))
	else
			Accuracy=$Accuracy
	fi

done
done

echo $Accuracy