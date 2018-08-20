// Databricks notebook source
import org.apache.spark.rdd._
import org.apache.spark.SparkContext._
import scala.collection.JavaConverters._
import au.com.bytecode.opencsv.CSVReader
import java.io._
import org.joda.time._
import org.joda.time.format._
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.feature.StandardScaler
import org.apache.spark.mllib.classification.{LogisticRegressionModel, LogisticRegressionWithLBFGS}
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.mllib.classification.SVMWithSGD
import org.apache.spark.mllib.tree.DecisionTree
import org.apache.spark.mllib.tree.RandomForest
import org.apache.spark.mllib.tree.model.RandomForestModel
import org.apache.spark.mllib.util.MLUtils

case class DelayRecord(
  year: String,
  month: String,
  dayOfMonth: String, 
  dayOfWeek: String, 
  departureTime: String,
  departureDelay: String, 
  originAirport: String, 
  distanceOfFlight: String,
  cancelled: String
){ 

  val listOfHolidays = List("01/01/2007", "01/15/2007", "02/19/2007", "05/28/2007", "06/07/2007", "07/04/2007",
        "09/03/2007", "10/08/2007" ,"11/11/2007", "11/22/2007", "12/25/2007",
        "01/01/2008", "01/21/2008", "02/18/2008", "05/22/2008", "05/26/2008", "07/04/2008",
        "09/01/2008", "10/13/2008" ,"11/11/2008", "11/27/2008", "12/25/2008")

  def generate_features: ((String,String), Array[Double]) = {
    val values = Array(
      departureDelay.toDouble,
      month.toDouble, 
      dayOfMonth.toDouble, 
      dayOfWeek.toDouble,
      get_hour(departureTime).toDouble, 
      distanceOfFlight.toDouble, 
      number_of_days_from_nearest_holiday(year.toInt, month.toInt, dayOfMonth.toInt)
    )

    //Creating a key-value pair, where we make the key as the date and the feature array is the value. 
    //The date will be used to combine it with the weather data. 
    new Tuple2((originAirport,get_date(year.toInt, month.toInt, dayOfMonth.toInt)), values)
  }

  def get_hour(departureTime: String): String = "%04d".format(departureTime.toInt).take(2)
  //def get_date(year: Int, month: Int, day: Int) = "%04d-%02d-%02d".format(year, month, day)
  def get_date(year: Int, month: Int, day: Int) = "%d/%d/%02d".format(month, day, year%100)

  def number_of_days_from_nearest_holiday(year:Int, month: Int, day: Int): Int = {
    val dateUnderConsideration = new DateTime(year, month, day, 0, 0)

    //foldLeft function applies the function with the current element of the collection and passes its result as the first parameter (here: r) to the next call of  
    // the function. The next call of the function will be on the next element of the collection. Here, 3000 is the initial value of r
    listOfHolidays.foldLeft(3000){ (r,c) => 
      val holidayDate = DateTimeFormat.forPattern("MM/dd/yyyy").parseDateTime(c)
      val difference = Math.abs(Days.daysBetween(holidayDate, dateUnderConsideration).getDays)
      math.min(r, difference)
    }
  }
}

//We filter the records for NOT-cancelled flights as we are interested in 'Delayed Flights' only and also filter records for specific airports. 
def preprocessing_Flight_Delay_Records(entireFile: String): RDD[DelayRecord] = {
  val data = sc.textFile(entireFile)
  
  data.map { line => 
    val reader = new CSVReader(new StringReader(line))
    reader.readAll().asScala.toList.map(record => DelayRecord(record(0), record(1), record(2), record(3), record(5), record(15), record(16), record(18), record(21)))
  }.map(list => list(0))
  .filter(record => record.year != "Year")
  .filter(record => record.cancelled == "0")
  .filter(record => record.originAirport == "ORD" | record.originAirport == "JFK" | record.originAirport == "ATL" | record.originAirport == "DFW" | record.originAirport == "LAX")
}

//If you see the last part, generate_features._2 it will be of the format Array[Double] and will contain the following parts: 
//departure Delay, month, day of the month, day of the week, hour of the departure time, distance of the flight, number of days from the nearest holiday.

val FlightData2007 = preprocessing_Flight_Delay_Records("/FileStore/tables/flight_dataset_2007.csv").map(record => record.generate_features._2)
val FlightData2008 = preprocessing_Flight_Delay_Records("/FileStore/tables/flight_dataset_2008.csv").map(record => record.generate_features._2)

System.out.println("Using only the Flight Data \n")
System.out.println("Training data features: \n departure Delay, month, day of the month, day of the week, hour of the departure time, distance of the flight, number of days from the nearest holiday \n ")
FlightData2007.take(5).map(x => x.mkString(",")).foreach(println)
 
//We consider flight delays of 15 minutes or more as delays and mark it with a label of 1.0, and under 15 minutes as non-delay and mark it with a label of 0.0.

def labelling_Data(vals: Array[Double]): LabeledPoint = {
  LabeledPoint(if (vals(0)>=15) 1.0 else 0.0, Vectors.dense(vals.drop(1)))
}

// Preparing the training set
//Ml-Lib uses Stochastic Gradient Descent which works best if the feature are normalized. Hence, we make use of the StandardScaler class for normalizing the  features. 
val labelledTrainingData = FlightData2007.map(labelling_Data)
labelledTrainingData.cache
val scalerForNormalization = new StandardScaler(withMean = true, withStd = true).fit(labelledTrainingData.map(x => x.features))
val normalizedTrainingData = labelledTrainingData.map(x => LabeledPoint(x.label, scalerForNormalization.transform(Vectors.dense(x.features.toArray))))
normalizedTrainingData.cache

// Preparing the test set
val labelledTestData = FlightData2008.map(labelling_Data)
labelledTestData.cache
val normalizedTestData = labelledTestData.map(x => LabeledPoint(x.label, scalerForNormalization.transform(Vectors.dense(x.features.toArray))))
normalizedTestData.cache

System.out.println(" \n Normalized training data with labels. Label 1 for flights with delay of more than 15 mins and Label 0 otherwise \n")
normalizedTrainingData.take(3).map(x => (x.label, x.features)).foreach(println)

// Metrics being computed: precision, recall, accuracy, and F1 measure. 
def metrics_evaluation(labelsAndPreds: RDD[(Double, Double)]) : Tuple2[Array[Double], Array[Double]] = {
    val true_positive = labelsAndPreds.filter(r => r._1==1 && r._2==1).count.toDouble
    val true_negative = labelsAndPreds.filter(r => r._1==0 && r._2==0).count.toDouble
    val false_postive = labelsAndPreds.filter(r => r._1==1 && r._2==0).count.toDouble
    val false_negative = labelsAndPreds.filter(r => r._1==0 && r._2==1).count.toDouble

    val precision = true_positive / (true_positive+false_postive)
    val recall = true_positive / (true_positive+false_negative)
    val F_measure = 2*precision*recall / (precision+recall)
    val accuracy = (true_positive+true_negative) / (true_positive+true_negative+false_postive+false_negative)
    new Tuple2(Array(true_positive, true_negative, false_postive, false_negative), Array(precision, recall, F_measure, accuracy))
}

// COMMAND ----------

def join_flight_and_weather_data(flight_data_file: String, weather_data_file: String): RDD[Array[Double]] = { 
  
  case class WeatherRecordClass(
  DATE: String,
  AWND: String,
  PRCP: String, 
  SNOW: String, 
  TMAX: String,
  TMIN: String,
  CODE: String  
  ){
    def generate_features_weather: ((String,String), Array[Double]) = {
      
      val values = Array(
        AWND.toDouble,
        PRCP.toDouble, 
        SNOW.toDouble, 
        TMAX.toDouble,
        TMIN.toDouble
      )
      //Creating a key-value pair, where we make the key as the date and the feature array is the value. 
      //The date will be used to combine it with the flight data.
      new Tuple2((CODE,DATE), values)
    }
  }
  
  def preprocessing_Weather_Records(entireFile: String): RDD[WeatherRecordClass] = {
    val data = sc.textFile(entireFile)

    data.map{ line => 
      val reader = new CSVReader(new StringReader(line))
      reader.readAll().asScala.toList.map(record => WeatherRecordClass(record(2), record(3), record(4), record(5), record(6), record(7), record(8)))
    }.map(list => list(0)).filter(record => record.DATE != "DATE") //filtering out the header. 
  }
  
  val weatherRecords= preprocessing_Weather_Records(weather_data_file).map{ record => 
        val features = record.generate_features_weather
        (features._1, features._2) //Tuple of date and features.
  }
  
  val flightRecords = preprocessing_Flight_Delay_Records(flight_data_file).map{ record => 
        val features = record.generate_features
        (features._1, features._2) //Tuple of date and features.     
  }
  
  weatherRecords.take(50).map(x => (x._1,x._2.mkString(","))).foreach(println)
  flightRecords.take(5).map(x => (x._1,x._2.mkString(","))).foreach(println)
  
  flightRecords.join(weatherRecords).map(vals => (vals._2._1 ++ vals._2._2))
}

val flight_weather_data_2007 = join_flight_and_weather_data("/FileStore/tables/flight_dataset_2007.csv", "/FileStore/tables/weather_dataset_2007_ALL.csv")
val flight_weather_data_2008 = join_flight_and_weather_data("/FileStore/tables/flight_dataset_2008.csv", "/FileStore/tables/weather_dataset_2008_ALL.csv")

println("\n Flight Weather Data 2007 \n")
flight_weather_data_2007.take(50).map(x => x.mkString(",")).foreach(println)

println("\n Flight Weather Data 2008 \n")
flight_weather_data_2008.take(50).map(x => x.mkString(",")).foreach(println)

// COMMAND ----------

//PREPARING TEST AND TRAINING DATA FOR FLIGHT + WEATHER DATASETS. 
def labeldata_fw(vals: Array[Double]): LabeledPoint = {
  LabeledPoint(if (vals(0)>=15) 1.0 else 0.0, Vectors.dense(vals.drop(1)))
}

// Prepare training set
val parsedTrainingData_fw = flight_weather_data_2007.map(labeldata_fw)
val scaler = new StandardScaler(withMean = true, withStd = true).fit(parsedTrainingData_fw.map(x => x.features))
val scaledTrainingData_fw = parsedTrainingData_fw.map(x => LabeledPoint(x.label, scaler.transform(Vectors.dense(x.features.toArray))))
parsedTrainingData_fw.cache
scaledTrainingData_fw.cache

// Prepare test/validation set
val parsedTestData_fw = flight_weather_data_2008.map(labeldata_fw)
val scaledTestData_fw = parsedTestData_fw.map(x => LabeledPoint(x.label, scaler.transform(Vectors.dense(x.features.toArray))))
parsedTestData_fw.cache
scaledTestData_fw.cache

scaledTrainingData_fw.take(5).map(x => (x.label, x.features)).foreach(println)


// COMMAND ----------

//ORIGINAL TRAINING DATA METRICS

//FLIGHT DATA - LOGISTIC REGRESSION
// Building a Logistic Regression model using original training data. 
val model_logistic_regression = new LogisticRegressionWithLBFGS().setNumClasses(2).run(labelledTrainingData)
// Predict
val predictedAndActualLabels_logistic_regression = labelledTestData.map { case LabeledPoint(label, features) =>
  val prediction = model_logistic_regression.predict(features)
  (prediction, label)
}

 
val (countsForLogisticRegression, metricsForLogisticRegression) = metrics_evaluation(predictedAndActualLabels_logistic_regression) 

println("\n Logistic Regression Metrics \n Using only flight data \n Accuracy = %.2f percent".format( metricsForLogisticRegression(3)*100))
println("\n True Positives = %.2f, True Negatives = %.2f, False Positives = %.2f, False Negatives = %.2f".format(countsForLogisticRegression(0), countsForLogisticRegression(1), countsForLogisticRegression(2), countsForLogisticRegression(3)))

println("\n Other metrics")
println(" precision = %.2f percent, recall = %.2f percent, F1 = %.2f percent".format(metricsForLogisticRegression(0)*100, metricsForLogisticRegression(1)*100, metricsForLogisticRegression(2)*100))

//FLIGHT + WEATHER DATA - LOGISTIC REGRESSION
//Before Normalization
val model_logistic_regression_fw = new LogisticRegressionWithLBFGS().setNumClasses(2).run(parsedTrainingData_fw)
// Predict
val predictedAndActualLabels_logistic_regression_fw = parsedTestData_fw.map { case LabeledPoint(label, features) =>
  val prediction = model_logistic_regression_fw.predict(features)
  (prediction, label)
}

val (countsForLogisticRegression_fw,metricsForLogisticRegression_fw) = metrics_evaluation(predictedAndActualLabels_logistic_regression_fw) 
println("\n Logistic Regression Metrics \n Using flight and weather data \n Accuracy = %.2f percent".format( metricsForLogisticRegression_fw(3)*100))
println("\n True Positives = %.2f, True Negatives = %.2f, False Positives = %.2f, False Negatives = %.2f".format(countsForLogisticRegression_fw(0), countsForLogisticRegression_fw(1), countsForLogisticRegression_fw(2), countsForLogisticRegression_fw(3)))

println("\n Other metrics")
println(" precision = %.2f percent, recall = %.2f percent, F1 = %.2f percent".format(metricsForLogisticRegression_fw(0)*100, metricsForLogisticRegression_fw(1)*100, metricsForLogisticRegression_fw(2)*100))

//FLIGHT DATA - SUPPORT VECTOR MACHINES
//Build the SVM model
val svmAlgorithm = new SVMWithSGD()
svmAlgorithm.optimizer.setNumIterations(100).setRegParam(1.0).setStepSize(1.0)
val model_svm = svmAlgorithm.run(labelledTrainingData)

// Predict
val predictedAndActualLabels_svm = labelledTestData.map {
  case LabeledPoint(label, features) =>
  val prediction = model_svm.predict(features)
  (prediction, label)
}

val (counts_svm, metrics_svm) = metrics_evaluation(predictedAndActualLabels_svm)

println("\n SVM Metrics \n using only flight data \n Accuracy = %.2f percent".format( metrics_svm(3)*100))
println("\n True Positives = %.2f, True Negatives = %.2f, False Positives = %.2f, False Negatives = %.2f".format(counts_svm(0), counts_svm(1), counts_svm(2), counts_svm(3)))

println("\n Other metrics")
println(" precision = %.2f percent, recall = %.2f percent, F1 = %.2f percent".format(metrics_svm(0)*100, metrics_svm(1)*100, metrics_svm(2)*100))



//FLIGHT + WEATHER DATA - SUPPORT VECTOR MACHINES
// Build the SVM model
val svmAlgorithm_fw = new SVMWithSGD()
svmAlgorithm_fw.optimizer.setNumIterations(100).setRegParam(1.0).setStepSize(1.0)
val model_svm_fw = svmAlgorithm_fw.run(parsedTrainingData_fw)

// Predict
val predictedAndActualLabels_svm_fw = parsedTestData_fw.map {
  case LabeledPoint(label, features) =>
  val prediction = model_svm_fw.predict(features)
  (prediction, label)
}

val (counts_svm_fw,metrics_svm_fw) = metrics_evaluation(predictedAndActualLabels_svm_fw)
println("\n SVM Metrics \n using flight and weather data \n Accuracy = %.2f percent".format( metrics_svm_fw(3)*100))
println("\n True Positives = %.2f, True Negatives = %.2f, False Positives = %.2f, False Negatives = %.2f".format(counts_svm_fw(0), counts_svm_fw(1), counts_svm_fw(2), counts_svm_fw(3)))

println("\n Other metrics")
println(" precision = %.2f percent, recall = %.2f percent, F1 = %.2f percent".format(metrics_svm_fw(0)*100, metrics_svm_fw(1)*100, metrics_svm_fw(2)*100))

// FLIGHT DATA - DECISION TREE
// Building the Decision Tree model
val numClasses = 2
val categoricalFeaturesInfo = Map[Int, Int]()
val impurity = "gini"
val maxDepth = 10
val maxBins = 100
val model_decision_tree = DecisionTree.trainClassifier(labelledTrainingData, numClasses, categoricalFeaturesInfo, impurity, maxDepth, maxBins)

// Prediction
val predictedAndActualLabels_decision_tree = labelledTestData.map { 
    case LabeledPoint(label, features) =>
    val prediction = model_decision_tree.predict(features)
    (prediction, label)
}

val (counts_decision_tree,metrics_decision_tree) = metrics_evaluation(predictedAndActualLabels_decision_tree)
println("\n Decision Tree Metrics \n using only flight data \n Accuracy = %.2f percent".format( metrics_decision_tree(3)*100))
println("\n True Positives = %.2f, True Negatives = %.2f, False Positives = %.2f, False Negatives = %.2f".format(counts_decision_tree(0), counts_decision_tree(1), counts_decision_tree(2), counts_decision_tree(3)))

println("\n Other metrics")
println(" precision = %.2f percent, recall = %.2f percent, F1 = %.2f percent".format(metrics_decision_tree(0)*100, metrics_decision_tree(1)*100, metrics_decision_tree(2)*100))


//FLIGHT + WEATHER DATA - DECISION TREE
val model_decision_tree_fw = DecisionTree.trainClassifier(parsedTrainingData_fw, numClasses, categoricalFeaturesInfo, impurity, maxDepth, maxBins)

// Prediction
val predictedAndActualLabels_decision_tree_fw = parsedTestData_fw.map { 
    case LabeledPoint(label, features) =>
    val prediction = model_decision_tree_fw.predict(features)
    (prediction, label)
}

val (counts_decision_tree_fw,metrics_decision_tree_fw) = metrics_evaluation(predictedAndActualLabels_decision_tree_fw)
println("\n Decision Tree Metrics \n using flight and weather data \n Accuracy = %.2f percent".format( metrics_decision_tree_fw(3)*100))
println("\n True Positives = %.2f, True Negatives = %.2f, False Positives = %.2f, False Negatives = %.2f".format(counts_decision_tree_fw(0), counts_decision_tree_fw(1), counts_decision_tree_fw(2), counts_decision_tree_fw(3)))

println("\n Other metrics")
println(" precision = %.2f percent, recall = %.2f percent, F1 = %.2f percent".format(metrics_decision_tree_fw(0)*100, metrics_decision_tree_fw(1)*100, metrics_decision_tree_fw(2)*100))

// COMMAND ----------

import org.apache.spark.mllib.tree.RandomForest
import org.apache.spark.mllib.tree.model.RandomForestModel
import org.apache.spark.mllib.util.MLUtils

// Train a RandomForest model.
// Empty categoricalFeaturesInfo indicates all features are continuous.
val numClasses = 2
val categoricalFeaturesInfo = Map[Int, Int]()
val numTrees = 10 // Use more in practice.
val featureSubsetStrategy = "auto" // Let the algorithm choose.
val impurity = "gini"
val maxDepth = 4
val maxBins = 32

//ORIGINAL TRAINING DATA
//FLIGHT DATA - RANDOM FORESTS
val model_random_forest = RandomForest.trainClassifier(labelledTrainingData, numClasses, categoricalFeaturesInfo,
  numTrees, featureSubsetStrategy, impurity, maxDepth, maxBins)

// Evaluate model on test instances and compute test error
val labelAndPreds_random_forest = labelledTestData.map { point =>
  val prediction = model_random_forest.predict(point.features)
  (point.label, prediction)
}

val (counts_random_forest, metrics_random_forest) = metrics_evaluation(labelAndPreds_random_forest)
println("\n Random Forest Metrics \n using flight data \n Accuracy = %.2f percent".format( metrics_random_forest(3)*100))
println("\n True Positives = %.2f, True Negatives = %.2f, False Positives = %.2f, False Negatives = %.2f".format(counts_random_forest(0), counts_random_forest(1), counts_random_forest(2), counts_random_forest(3)))

println("\n Other metrics")
println(" precision = %.2f percent, recall = %.2f percent, F1 = %.2f percent".format(metrics_random_forest(0)*100, metrics_random_forest(1)*100, metrics_random_forest(2)*100))

//ORIGINAL TRAINING DATA
//FLIGHT + WEATHER DATA - RANDOM FORESTS
val model_random_forest_fw = RandomForest.trainClassifier(parsedTrainingData_fw, numClasses, categoricalFeaturesInfo,
  numTrees, featureSubsetStrategy, impurity, maxDepth, maxBins)

// Evaluate model on test instances and compute test error
val labelAndPreds_random_forest_fw = parsedTestData_fw.map { point =>
  val prediction = model_random_forest_fw.predict(point.features)
  (point.label, prediction)
}

val (counts_random_forest_fw, metrics_random_forest_fw) = metrics_evaluation(labelAndPreds_random_forest_fw)
println("\n Random Forest Metrics \n using flight and weather data data \n Accuracy = %.2f percent".format( metrics_random_forest_fw(3)*100))
println("\n True Positives = %.2f, True Negatives = %.2f, False Positives = %.2f, False Negatives = %.2f".format(counts_random_forest_fw(0), counts_random_forest_fw(1), counts_random_forest_fw(2), counts_random_forest_fw(3)))

println("\n Other metrics")
println(" precision = %.2f percent, recall = %.2f percent, F1 = %.2f percent \n".format(metrics_random_forest_fw(0)*100, metrics_random_forest_fw(1)*100, metrics_random_forest_fw(2)*100))


// COMMAND ----------

//NORMALIZED TRAINING DATA METRICS - ONLY FLIGHT DATA

// Building a Logistic Regression model using the normalized training data. 
val model_logistic_regression = new LogisticRegressionWithLBFGS().setNumClasses(2).run(normalizedTrainingData)
// Predict
val predictedAndActualLabels_logistic_regression = normalizedTestData.map { case LabeledPoint(label, features) =>
  val prediction = model_logistic_regression.predict(features)
  (prediction, label)
}

val metricsForLogisticRegression = metrics_evaluation(predictedAndActualLabels_logistic_regression)._2 //just taking the array containing precision, recall, F_measure, accuracy
println("\n Logistic Regression Metrics \n precision = %.2f, recall = %.2f, F1 = %.2f, accuracy = %.2f \n".format(metricsForLogisticRegression(0), metricsForLogisticRegression(1), metricsForLogisticRegression(2), metricsForLogisticRegression(3)))

// Build the SVM model
val svmAlgorithm = new SVMWithSGD()
svmAlgorithm.optimizer.setNumIterations(100).setRegParam(1.0).setStepSize(1.0)
val model_svm = svmAlgorithm.run(normalizedTrainingData)

// Predict
val predictedAndActualLabels_svm = normalizedTestData.map {
  case LabeledPoint(label, features) =>
  val prediction = model_svm.predict(features)
  (prediction, label)
}

val metrics_svm = metrics_evaluation(predictedAndActualLabels_svm)._2
println("\n SVM Metrics: \n precision = %.2f, recall = %.2f, F1 = %.2f, accuracy = %.2f \n ".format(metrics_svm(0), metrics_svm(1), metrics_svm(2), metrics_svm(3)))

// Building the Decision Tree model
val numClasses = 2
val categoricalFeaturesInfo = Map[Int, Int]()
val impurity = "gini"
val maxDepth = 10
val maxBins = 100
val model_decision_tree = DecisionTree.trainClassifier(normalizedTrainingData, numClasses, categoricalFeaturesInfo, impurity, maxDepth, maxBins)

// Prediction
val predictedAndActualLabels_decision_tree = normalizedTestData.map { 
    case LabeledPoint(label, features) =>
    val prediction = model_decision_tree.predict(features)
    (prediction, label)
}

val metrics_decision_tree = metrics_evaluation(predictedAndActualLabels_decision_tree)._2
println("\n Decision Tree Metrics: \n precision = %.2f, recall = %.2f, F1 = %.2f, accuracy = %.2f \n".format(metrics_decision_tree(0), metrics_decision_tree(1), metrics_decision_tree(2), metrics_decision_tree(3)))

//Building a Naive Bayes Model - Can't use this because for Naive Bayes only non-negative values work. 
// val model_Naive_Bayes = NaiveBayes.train(normalizedTrainingData, lambda = 1.0, modelType = "multinomial")

// val predictedAndActualLabels_Naive_Bayes = normalizedTestData.map{ 
//     case LabeledPoint(label, features) =>
//     val prediction = model_decision_tree.predict(features)
//     (prediction, label)
// }

// val metrics_Naive_Bayes = metrics_evaluation(predictedAndActualLabels_Naive_Bayes)._2
// println("\n Naive Bayes Metrics: \n precision = %.2f, recall = %.2f, F1 = %.2f, accuracy = %.2f \n".format(metrics_Naive_Bayes(0), metrics_Naive_Bayes(1), metrics_Naive_Bayes(2), metrics_Naive_Bayes(3)))

