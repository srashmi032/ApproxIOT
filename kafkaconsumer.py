from __future__ import print_function  # python 2/3 compatibility
 
import sys # used to exit
from kafka import KafkaConsumer


import random
import time
import csv


csv.register_dialect('myDialect',
quoting=csv.QUOTE_ALL,
skipinitialspace=True)
print ("rash") 
KAFKA_TOPIC = 'sample'
topic2 = 'test'
KAFKA_BROKERS = 'localhost:9092' # see step 1
 
consumer = KafkaConsumer(KAFKA_TOPIC, bootstrap_servers=KAFKA_BROKERS, )
#length=KafkaConsumer(KAFKA_TOPIC_SIZE, bootstrap_servers=KAFKA_BROKERS, )
consumer2=KafkaConsumer(topic2, bootstrap_servers=KAFKA_BROKERS, )


data=[]
k=5
avg_size=0
tot_size=0
number=0;
start_time = time.time()

for message in consumer2:
	if len(data)<k:
		data.append(message.value)
		print (sys.getsizeof(message))
		tot_size+=sys.getsizeof(message)
		number+=1
	#print (len(data))
	elif k==len(data):
		print ("new reservoir:")
		with open('sample_data_abc.csv','w') as f:
			writer=csv.writer(f,dialect='myDialect')
			for x in range(0,len(data)):
				print (data[x])
				#list1=data[x].split(',')
				f.write(','.join(data[x].split(',')))
		f.close()
		

		index=random.randrange(k);
		data[index]=message.value 
		avg_size=tot_size/k

		time_taken=time.time()-start_time

		print ("time:",time_taken)

		print ("Processed {0} messages in {1} seconds".format(k,time_taken))
		throughput=k/time_taken

		print ("throughput in Msgs/s:",throughput)

		break


data=[]
k=10000
avg_size=0
tot_size=0
number=0;
start_time = time.time()
for message in consumer:
	if len(data)<k:
		data.append(message.value)
		print (sys.getsizeof(message))
		tot_size+=sys.getsizeof(message)
		number+=1
	#print (len(data))
	elif k==len(data):
		print ("new reservoir:")
		with open('sample_data.csv','w') as f:
			writer=csv.writer(f,dialect='myDialect')
			for x in range(0,len(data)):
				print (data[x])
				#list1=data[x].split(',')
				f.write(','.join(data[x].split(',')))
		f.close()
		
			#consumer.stop()




		index=random.randrange(k);
		data[index]=message.value 
		avg_size=tot_size/k

		time_taken=time.time()-start_time

		print ("time:",time_taken)

		print ("Processed {0} messages in {1} seconds".format(k,time_taken))
		throughput=k/time_taken

		print ("throughput in Msgs/s:",throughput)

		break

	
		



	



	
	






	



#except KeyboardInterrupt:
    #sys.exit()

