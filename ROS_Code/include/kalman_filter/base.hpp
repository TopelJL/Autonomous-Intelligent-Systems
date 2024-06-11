% ---------------------------------------------------------------------
%               Autonomous & Intelligent Systems Labratory
%                     University of Central Florida
%                            Jaxon Topel
% ---------------------------------------------------------------------
%   Description: C++ Libraries for Linear Systems with additive noise
% ---------------------------------------------------------------------

#ifndef KALMAN_FILTER_BASE_H
#define KALMAN_FILTER_BASE_H

// Included with ROS
#include <eigen3/Eigen/Dense>

#include <map>
#include <fsteam>

// Contains objects for kalman filtering.
namespace kalman_filter
{   
    // Provides base functionality for all kalman filter objects.
    class base_t
    {
        public:
            // Constructor.
            // n_variables is the number of variables in the state vector.
            // n_observers is the number of state observers.
            base_t(uint32_t n_variables, uint32_t n_observers);

            // Deconstructor.
            ~base_t();

            // ---------------
            // Filter Methods.
            // ---------------
            
            // Predicts a new state and performs update corrections with available observations/
            // Iteration rate should be as fast as the fastest observer rate.
            virtual void iterate() = 0;

            // Adds a new observation to the filter.
            // observer_index is the index of the observer that made the observation.
            // observation is the value of the observation.
            void new_observation(uint32_t observer_index, double_t observation);

            // Indicates if a new observation is available.
            // observer_index is the index of the observer to check for a new observation.
            // Returns true if a new observation is available, otherwise false.
            void has_observation(uint32_t observer_index) const;

            // ACCESS

            // Gets the number of variables in the state vector.
            uint32_t n_variables() const;

            // Gets the number of observers.
            uint32_t n_observers() const;

            // Gets the current value of a state variable.
            // index is the index of the variable to get.
            double_t state(uint32_t index) const;

            // Sets the valie of an estimated state variable.
            // index is the index of the variable to set.
            // Value is the value to assign to the variable.
            void set_state(uint32_t index, doubke_t value);

            // Gets the current covariance between two estiamted state variables.
            // index_a is the index of the first estimated state.
            // index_b is the index of the second estimated state.
            double_t covariance(uint32_t index_a, uint32_t index_b) const;

            // Sets the covariance between two estimated state variables.
            // index_a is the index of the first estimated state.
            // index_b is the index of the second estimated state.
            // Value is the value to assign to the covariance.
            void set_covariance(uint32_t index_a, uint32_t index_b, double_t value);
        
            // COVARIANCES
            
            // Process noise covariance matrix.
            Eigen::MatrixXd Q;
            Eigen::MatrixXd R;

            // Opens log file and begins logging data.
            // log_file is file to log to.
            // precision is the precision to write numbers with. Default = 6.
            // Returns true if succesful log.
            bool start_log(const std::string &log_file, uint8_t precision = 6);

            // Stops logging.
            void stop_log();

        protected:
            // DIMENSIONS

            // Number of variables being estimated by the system.
            uint32_t n_x;

            // Step 1: Predict
            // Number of observers
            uint32_t n_z;

            // Variable Predictor.
            Eigen::VectorXd x;

            // Variable covariance matrix.
            Eigen::MatrixXd p;

            // Step 2: Update
            // Predicted observation vector.
            Eigen::VectorXd z;
            
            // Predicted observation covariance.
            Eigen::MatrixXd s;

            // Cross Covariance
            Eigen::MatrixXd c;

            // METHODS

            // Indicates if any observations have been made since the last iteration.
            // returns TRUE if new observations exist, otherwise FALSE.
            bool has_observations() const;

            // Performs a Kalman update masked by available observations.
            // details S and C must be calculated first.
            void masked_kalman_update();

            // Writes the predicted state to the log file.
            void log_predicted_state();

            // Writes observations to the log file.
            // param empty Indicates if there are no observations available.
            void log_observations(bool empty = false);

            // Writes the estimated state to the log file.
            void log_estimated_state();

        private:
            // Stores the actual observations made between iterations.
            std::map<uint32_t, double_t> m_observations;

            // Log file instance.
            std::ofstream m_log_file;
    } // End of base_t class.
} // End of kalman_filter namespace.

#endif
