from kafka import KafkaProducer
import sys
import time




KAFKA_TOPIC = 'sample'
topic2='test'

KAFKA_BROKERS = 'localhost:9092' # see step 1
 
producer=KafkaProducer(bootstrap_servers=KAFKA_BROKERS)
producer2=KafkaProducer(bootstrap_servers=KAFKA_BROKERS)
# Must send bytes
#messages = [b'hello kafka', b'I am sending', b'3 test messages']
 
# Send the messages
file=open("taxi copy.csv","r")
file2=open("abc.rtf","r")
number=0;

avg_size=0
tot_size=0

start_time = time.time()

for m in file2:
	number=number+1
	producer.send(topic2, m).get(timeout=10)
	print (sys.getsizeof(m))
	tot_size+=sys.getsizeof(m)

print (number)
print (tot_size/number)
avg_size=tot_size/number

time_taken=time.time()-start_time

print ("time:",time_taken)

print ("Processed {0} messages in {1} seconds".format(number,time_taken))
throughput=number/time_taken

print ("throughput in Msgs/s:",throughput)


number=0;
avg_size=0
tot_size=0

start_time = time.time()
for m in file:
	number=number+1
	producer.send(KAFKA_TOPIC, m).get(timeout=10)
	print (sys.getsizeof(m))
	tot_size+=sys.getsizeof(m)


print (number)
print (tot_size/number)
avg_size=tot_size/number

time_taken=time.time()-start_time

print ("time:",time_taken)

print ("Processed {0} messages in {1} seconds".format(number,time_taken))
throughput=number/time_taken

print ("throughput in Msgs/s:",throughput)

#producer.send(KAFKA_TOPIC, number).get(timeout=10)
