package com.demo.spark


import scala.collection.mutable._
import org.apache.log4j._
import org.apache.spark._
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SparkSession

import scala.collection.mutable

object sp_rst {

  val pointers = Map("Patient ID" -> 0, "Headache" -> 1, "Musclepain" -> 2, "Temperature" -> 3, "Flu" -> 4)


  case class Dataschema(Id: String, Headache: String, Musclepain: String, Temperature: String, Flu: String)


  // ALGORITHM 2 (calculateIND)
  def calculateIND(D: Array[Array[String]], no_of_attributes: Int, tuples: Int): Map[String, List[String]] = {
    var IND = scala.collection.mutable.Map[String, List[String]]();
    var i = 0;
    var j = 0;
    for (i <- 0 until tuples) {
      var value = D(i)(no_of_attributes - 1);
      var all = new ListBuffer[String]();
      if (IND.contains(value)) {
        for (j <- 0 until IND(value).length) {
          all += IND(value)(j)
        }
      }
      all += D(i)(0);
      IND(value) = all.toList;
    }
    return IND;
  }

//  ALGORITHM
//  3

  def generateAllComb(D: Array[Array[String]], start: Int, end: Int, no_of_attributes: Int, tuples: Int, all_attributes: Array[String]): List[List[String]] = {
    var attributes = new ListBuffer[String]()
    for (i <- start to end) {
      attributes += all_attributes(i);
    }
    var allcomb_temp = attributes.toList.toSet[String].subsets.map(_.toList).toList;
    var allcomb1 = new ListBuffer[List[String]]()
    for (i <- 1 until allcomb_temp.length) {
      allcomb1 += allcomb_temp(i);
    }
    var allcomb = allcomb1.toList
    return allcomb;
  }

  // ALGORITHM 4 (IND)
  def calculateIND_(D: Array[Array[String]], start: Int, size: Int, no_of_attributes: Int, tuples: Int, all_attributes: Array[String], allcomb: List[List[String]]): Map[List[String], List[List[String]]] = {
    var IND_ = scala.collection.mutable.Map[List[String], List[List[String]]]();
    for (i <- 0 until allcomb.length) {
      var end = start;
      if (start + size < tuples) {
        end = start + size - 1;
      } else {
        end = tuples - 1;
      }
      var mapp = scala.collection.mutable.Map[ListBuffer[String], List[String]]();
      for (j <- start to end) {
        var all = new ListBuffer[String]();
        for (k <- 0 until allcomb(i).length) {
          all += D(j)(pointers(allcomb(i)(k)))
        }
        var all1 = new ListBuffer[String]();
        if (mapp.contains(all)) {
          for (k <- 0 until mapp(all).length) {
            all1 += mapp(all)(k)
          }
        }
        all1 += D(j)(0);
        mapp(all) = all1.toList;
      }
      var res = new ListBuffer[List[String]]();
      mapp.foreach { keyVal => res += keyVal._2 };
      var res1 = res.toList;
      IND_(allcomb(i)) = res1;
    }
    return IND_;
  }

  // ALGORITHM 5
  def generateDEP(allcomb: ListBuffer[List[List[String]]]  , IND_ALL :ListBuffer[Map[List[String], List[List[String]]]]  , IND:Map[String, List[String]]): Map[List[String], Int] = {
    val dep = scala.collection.mutable.Map[List[String], Int]();
    for (i <- 0 until allcomb.length) {
      for (j <- 0 until allcomb(i).length) {
        dep(allcomb(i)(j)) = 0;
      }
    }
    IND.foreach { keyVal => {
      var classs = keyVal._2;
      for (i <- 0 until IND_ALL.length) {
        IND_ALL(i).foreach { keyVal1 => {
          var list = keyVal1._2
          for (j <- 0 until list.length) {
            var exists = true
            for (k <- 0 until list(j).length) {
              if (!classs.contains(list(j)(k))) {
                exists = false
              }
            }
            if (exists) {
              dep(keyVal1._1) += 1
            }
          }
        }
        };
      }
    }
    }
    return dep;
  }

  // ALGORITHM 6
  def maxDEP(allcomb:ListBuffer[List[List[String]]]   , DEP:Map[List[String], Int]): List[Int] = {
    val rr = new ListBuffer[Int]();
    for (i <- 0 until allcomb.length) {
      var max = 0;
      for (j <- 0 until allcomb(i).length) {
        if (DEP(allcomb(i)(j)) > max) {
          max = DEP(allcomb(i)(j));
        }
      }
      rr += max;
    }
    return rr.toList
  }

  // ALGORITHM 7
  def filterDEP(allcomb:ListBuffer[List[List[String]]]  ,  DEP:Map[List[String], Int] , max_dep: List[Int] ): List[List[List[String]]] = {
    val final1 = new ListBuffer[List[List[String]]];
    for (i <- 0 until allcomb.length) {
      val f1 = new ListBuffer[List[String]];
      for (j <- 0 until allcomb(i).length) {
        if (DEP(allcomb(i)(j)) == max_dep(i)) {
          f1 += allcomb(i)(j)
        }
      }
      final1 += f1.toList
    }
    return final1.toList;
  }

  // ALGORITHM 8
  def minNBF(filtered_attributes : List[List[List[String]]]): List[List[String]] = {
    val reduced_attributes = new ListBuffer[List[String]];
    for (i <- 0 until filtered_attributes.length) {
      var min = 10000000;
      for (j <- 0 until filtered_attributes(i).length) {
        if (min > filtered_attributes(i)(j).length) {
          min = filtered_attributes(i)(j).length
        }
      }
      for (j <- 0 until filtered_attributes(i).length) {
        if (min == filtered_attributes(i)(j).length) {
          reduced_attributes += filtered_attributes(i)(j)
        }
      }
    }
    return reduced_attributes.toList;
  }


  def main(args: Array[String]) {
    Logger.getLogger("org").setLevel(Level.ERROR)

    // Create a SparkSession using every core of the local machine
    val spark = SparkSession
      .builder
      .appName("Testing")
      .master("local[*]")
      .getOrCreate()

    // Load each line of the source data into an Dataset
    import spark.implicits._
    val ds = spark.read
      .option("header", "true")
      .option("inferSchema", "true")
      .csv("data/sp_rstdata.csv")
      .as[Dataschema]


    val no_of_attributes = ds.columns.size
    val tuples = ds.count().intValue()

    println(tuples);
    println(no_of_attributes)
    val D = Array.ofDim[String](tuples, no_of_attributes);


    val all_attributes = Array("Id", "Headache", "Musclepain", "Temperature", "Flu");
    val decision_attribute = "Flu"

    val m = 2;
    val r = Array(1, 2);
    var allcomb = new ListBuffer[List[List[String]]]();
    var start = 1;

    var k = 0;

    for (i <- 0 until all_attributes.length) {

      println(i)
      val name = all_attributes(i)
      val curr = ds.select(s"${name}").map(_.getString(0)).collect().toList
      println(curr)

      for (j <- 0 until curr.length) {
        D(j)(k) = curr(j)

      }
      k = k + 1;

    }

    D.foreach { row => row foreach print; println }

    var IND = calculateIND(D, no_of_attributes, tuples);
    println("OUTPUT OF ALGORITHM 2 (INDISCERNIBILITY OF DECISION ATTRIBUTE SET): ")
      println(IND)
    println()



    for (i <- 0 until r.length) {
      allcomb += generateAllComb(D, start, start + r(i) - 1, no_of_attributes, tuples, all_attributes);
      start += r(i);
    }
    println("OUTPUT OF ALGORITHM 3 (ALL COMBINATIONS): ")
    println(allcomb.toList)
    println()
     var IND_ALL = new ListBuffer[Map[List[String], List[List[String]]]];
    start = 0;
    for (j <- 0 until r.length) {
      start = 0;
      while (start < tuples) {
        var ind = calculateIND_(D, start, tuples / m, no_of_attributes, tuples, all_attributes, allcomb(j));
        start += (tuples / m);
        IND_ALL += ind;
      }
    }
    println("OUTPUT OF ALGORITHM 4 (INDISCERNIBILITY OF ALL COMBINATIONS): ")
    for (i <- 0 until IND_ALL.length) {
      IND_ALL(i).foreach { keyVal => {
        println(keyVal._1 + " -> " + keyVal._2)
      }
      }
    }
    println()
    val DEP = generateDEP(allcomb , IND_ALL , IND );
    println("OUTPUT OF ALGORITHM 5 (DEPENDENCY MEASURES): ")
    println(DEP)
    println()
    val max_dep = maxDEP(allcomb , DEP);
    println("OUTPUT OF ALGORITHM 6 (MAXIMUM DEPENDENCY VALUE): ")
    println(max_dep)
    println()
    var filtered_attributes = filterDEP(allcomb  ,DEP ,max_dep );
    println("OUTPUT OF ALGORITHM 7 (FILTERED ATTRIBUTES USING BASELINE): ")
    println(filtered_attributes)
    println()
    val reduced_attribute_set = minNBF(filtered_attributes);
    println("FINAL REDUCED ATTRIBUTE SET: ")
    for (i <- 0 until reduced_attribute_set.length) {
      println(reduced_attribute_set(i));
    }
  }

}