Spark application:
	Spark Applications consist of a driver process and a set of executor processes. The driver process runs
	your main() function, sits on a node in the cluster, and is responsible for three things:
		1. maintaining information about the Spark Application
		2. responding to a user's program or input
		3. analyzing, distributing and scheduling work across the executors
		
	Driver is the heart of spark, it maintains all the information during the lifetime of spark application
	
	The executors are responsible for actually carrying out the work that the driver assigns them
	Executors are responsible for two things
		1. executing the code assigned to it.
		2. reporting the state of the computation on that executor back to the driver node.
	
Key Points about Spark Application
	1. Spark employs a cluster manager that keeps track of the resources available
	2. The driver process is responsible for executing the driver program's commands across the executors to complete a given task
The executor for the most part will always be running spark code