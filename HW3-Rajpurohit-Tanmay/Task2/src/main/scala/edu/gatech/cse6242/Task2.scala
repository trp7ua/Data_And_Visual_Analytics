package edu.gatech.cse6242

import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.SparkConf

object Task2 {
  def main(args: Array[String]) {
    val sc = new SparkContext(new SparkConf().setAppName("Task2"))

    // read the file
    val file = sc.textFile("hdfs://localhost:8020" + args(0))

// Split into Key/Value Pairs
    val data1 = file.map(x => (x.split("\\s")(0), x.split("\\s")(2)))
    val data2 = file.map(x => (x.split("\\s")(1), x.split("\\s")(2)))

// Change the Values according to the weights
    val data1T1 = data1.mapValues(x=>0.8*x.toInt) 
    val data2T1 = data2.mapValues(x=>0.2*x.toInt)

// Merge two dataset 
    val dataT2 = data1T1.union(data2T1)

// Reduce by key
    val output = dataT2.reduceByKey(_ + _);

// Storing the output
    output.saveAsTextFile("hdfs://localhost:8020" + args(1))
  }
}