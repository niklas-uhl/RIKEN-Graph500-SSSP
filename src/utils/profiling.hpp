/*
 * profiling.hpp
 *
 *  Created on: Jun 8, 2022
 *      Author: Daniel Rehfeldt
 */

#ifndef SRC_UTILS_PROFILING_HPP_
#define SRC_UTILS_PROFILING_HPP_

#include "utils.hpp"
#include <cstdio>
#include "parameters.h"

namespace profiling {

class ProfilingInformationStore {
public:
   void submit(double span, const char* content, int number) {
#pragma omp critical (pis_submit_time)
      times_.push_back(TimeElement(span, content, number));
   }
   void submit(int64_t span_micro, const char* content, int number) {
#pragma omp critical (pis_submit_time)
      times_.push_back(TimeElement((double)span_micro / 1000000.0, content, number));
   }
   void submitCounter(int64_t counter, const char* content, int number) {
#pragma omp critical (pis_submit_counter)
      counters_.push_back(CountElement(counter, content, number));
   }
   void reset() {
      times_.clear();
      counters_.clear();
   }
   void printResult() {
      printTimeResult();
      printCountResult();
   }
private:
   struct TimeElement {
      double span;
      const char* content;
      int number;

      TimeElement(double span__, const char* content__, int number__)
         : span(span__), content(content__), number(number__) { }
   };
   struct CountElement {
      int64_t count;
      const char* content;
      int number;

      CountElement(int64_t count__, const char* content__, int number__)
         : count(count__), content(content__), number(number__) { }
   };

   void printTimeResult() {
      int num_times = times_.size();
      double *dbl_times = new double[num_times];
      double *sum_times = new double[num_times];
      double *max_times = new double[num_times];

      for(int i = 0; i < num_times; ++i) {
         dbl_times[i] = times_[i].span;
      }

      MPI_Reduce(dbl_times, sum_times, num_times, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
      MPI_Reduce(dbl_times, max_times, num_times, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

      if(mpi.isMaster()) {
         for(int i = 0; i < num_times; ++i) {
            fprintf(IMD_OUT, "Time of %s, %d, Avg, %f, Max, %f, (ms)\n", times_[i].content,
                  times_[i].number,
                  sum_times[i] / mpi.size_2d * 1000.0,
                  max_times[i] * 1000.0);
         }
      }

      delete [] dbl_times;
      delete [] sum_times;
      delete [] max_times;
   }

   double displayValue(int64_t value) {
      if(value < int64_t(1000))
         return (double)value;
      else if(value < int64_t(1000)*1000)
         return value / 1000.0;
      else if(value < int64_t(1000)*1000*1000)
         return value / (1000.0*1000);
      else if(value < int64_t(1000)*1000*1000*1000)
         return value / (1000.0*1000*1000);
      else
         return value / (1000.0*1000*1000*1000);
   }

   const char* displaySuffix(int64_t value) {
      if(value < int64_t(1000))
         return "";
      else if(value < int64_t(1000)*1000)
         return "K";
      else if(value < int64_t(1000)*1000*1000)
         return "M";
      else if(value < int64_t(1000)*1000*1000*1000)
         return "G";
      else
         return "T";
   }

   void printCountResult() {
      int num_times = counters_.size();
      int64_t *dbl_times = new int64_t[num_times];
      int64_t *sum_times = new int64_t[num_times];
      int64_t *max_times = new int64_t[num_times];

      for(int i = 0; i < num_times; ++i) {
         dbl_times[i] = counters_[i].count;
      }

      MPI_Reduce(dbl_times, sum_times, num_times, MPI_INT64_T, MPI_SUM, 0, MPI_COMM_WORLD);
      MPI_Reduce(dbl_times, max_times, num_times, MPI_INT64_T, MPI_MAX, 0, MPI_COMM_WORLD);

      if(mpi.isMaster()) {
         for(int i = 0; i < num_times; ++i) {
            int64_t sum = sum_times[i], avg = sum_times[i] / mpi.size_2d, maximum = max_times[i];
            fprintf(IMD_OUT, "%s, %d, Sum, %ld, Avg, %ld, Max, %ld\n", counters_[i].content,
                  counters_[i].number, sum, avg, maximum);
         }
      }

      delete [] dbl_times;
      delete [] sum_times;
      delete [] max_times;
   }

   std::vector<TimeElement> times_;
   std::vector<CountElement> counters_;
};

ProfilingInformationStore g_pis;

class TimeKeeper {
public:
   TimeKeeper() : start_(get_time_in_microsecond()){ }
   void submit(const char* content, int number) {
      int64_t end = get_time_in_microsecond();
      g_pis.submit(end - start_, content, number);
      start_ = end;
   }
   int64_t getSpanAndReset() {
      int64_t end = get_time_in_microsecond();
      int64_t span = end - start_;
      start_ = end;
      return span;
   }
private:
   int64_t start_;
};

class TimeSpan {
   TimeSpan(int64_t init) : span_(init) { }
public:
   TimeSpan() : span_(0) { }
   TimeSpan(TimeKeeper& keeper) : span_(keeper.getSpanAndReset()) { }

   void reset() { span_ = 0; }
   TimeSpan& operator += (TimeKeeper& keeper) {
      __sync_fetch_and_add(&span_, keeper.getSpanAndReset());
      return *this;
   }
   TimeSpan& operator -= (TimeKeeper& keeper) {
      __sync_fetch_and_add(&span_, - keeper.getSpanAndReset());
      return *this;
   }
   TimeSpan& operator += (TimeSpan span) {
      __sync_fetch_and_add(&span_, span.span_);
      return *this;
   }
   TimeSpan& operator -= (TimeSpan span) {
      __sync_fetch_and_add(&span_, - span.span_);
      return *this;
   }
   TimeSpan& operator += (int64_t span) {
      __sync_fetch_and_add(&span_, span);
      return *this;
   }
   TimeSpan& operator -= (int64_t span) {
      __sync_fetch_and_add(&span_, - span);
      return *this;
   }

   TimeSpan operator + (TimeSpan span) {
      return TimeSpan(span_ + span.span_);
   }
   TimeSpan operator - (TimeSpan span) {
      return TimeSpan(span_ - span.span_);
   }

   void submit(const char* content, int number) {
      g_pis.submit(span_, content, number);
      span_ = 0;
   }
   double getSpan() {
      return (double)span_ / 1000000.0;
   }
private:
   int64_t span_;
};

volatile double expand_time;
volatile double expand_buckets_time;
volatile double expand_settled_bitmap_time;
volatile double fold_time;

} // namespace profiling



#endif /* SRC_UTILS_PROFILING_HPP_ */
