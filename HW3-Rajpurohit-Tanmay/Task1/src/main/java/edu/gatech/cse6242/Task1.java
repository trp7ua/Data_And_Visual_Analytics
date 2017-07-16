package edu.gatech.cse6242;

import java.io.IOException;
import java.util.*;

import org.apache.hadoop.fs.Path;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.io.*;
import org.apache.hadoop.mapreduce.*;
import org.apache.hadoop.util.*;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;

public class Task1 {

	public static class TokenizerMapper
    extends Mapper<LongWritable, Text, Text, IntWritable>{

 public void map(LongWritable key, Text value, Context context
                 ) throws IOException, InterruptedException {
 
	 String line = value.toString(); 
     String lasttoken = null; 
     StringTokenizer s = new StringTokenizer(line,"\t"); 
     String node = s.nextToken(); 
     
     while(s.hasMoreTokens())
     {
        lasttoken=s.nextToken();
     } 
     
     int weight = Integer.parseInt(lasttoken); 
     context.write(new Text(node),  new IntWritable(weight));
   }
  }

public static class IntSumReducer
    extends Reducer<Text,IntWritable,Text,IntWritable> {
	
 private IntWritable result = new IntWritable();
 public void reduce(Text key, Iterable<IntWritable> values,
                    Context context
                    ) throws IOException, InterruptedException {
   int max = 0;
   for (IntWritable val : values) {
     if(val.get()>max)
    	 max=val.get();
   }
   result.set(max);
   context.write(key, result);
 }
}

	
	
  public static void main(String[] args) throws Exception {
    Configuration conf = new Configuration();
    Job job = Job.getInstance(conf, "Task1");

    job.setJarByClass(Task1.class);
    
    job.setMapperClass(TokenizerMapper.class);
    job.setCombinerClass(IntSumReducer.class);
    job.setReducerClass(IntSumReducer.class);
    job.setMapOutputKeyClass(Text.class);
    job.setMapOutputValueClass(IntWritable.class);
    job.setOutputKeyClass(IntWritable.class);
    job.setOutputValueClass(IntWritable.class);  
    FileInputFormat.addInputPath(job, new Path(args[0]));
    FileOutputFormat.setOutputPath(job, new Path(args[1]));
    System.exit(job.waitForCompletion(true) ? 0 : 1);
  }
}