/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).
	num_particles = 1000;

	double std_x = std[0];
	double std_y = std[1];
	double std_theta = std[2];
	default_random_engine gen;

	normal_distribution<double> dist_x(x, std_x);
	normal_distribution<double> dist_y(y, std_y);
	normal_distribution<double> dist_theta(theta, std_theta);

	for (int i = 0; i < num_particles; ++i) {
		 Particle p;

		 p.x = dist_x(gen);
		 p.y = dist_y(gen);
		 p.theta = dist_theta(gen);
		 p.weight = 1.0;

		 particles.push_back(p);
	}

	is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/
	double std_x = std_pos[0];
	double std_y = std_pos[1];
	double std_theta = std_pos[2];
	default_random_engine gen;

	for (auto& p : particles) {
		if (fabs(yaw_rate) > 0.0001) {
			p.x += (velocity / yaw_rate) * (sin(p.theta + yaw_rate * delta_t) - sin(p.theta));
			p.y += (velocity / yaw_rate) * (cos(p.theta) - cos(p.theta + yaw_rate * delta_t));
		} else {
			p.x += velocity * cos(p.theta) * delta_t;
			p.y += velocity * sin(p.theta) * delta_t;
		}
		p.theta += yaw_rate * delta_t;

		normal_distribution<double> dist_x(p.x, std_x);
		normal_distribution<double> dist_y(p.y, std_y);
		normal_distribution<double> dist_theta(p.theta, std_theta);

		p.x = dist_x(gen);
		p.y = dist_y(gen);
		p.theta = dist_theta(gen);
	}
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.
	for (auto& obs : observations) {
		int id = predicted[0].id;
		double nearest_dist = dist(obs.x, obs.y, predicted[0].x, predicted[0].y);
		for (int i = 1; i < predicted.size(); i++) {
			double d = dist(obs.x, obs.y, predicted[i].x, predicted[i].y);
			if (d < nearest_dist) {
				id = predicted[i].id;
				nearest_dist = d;
			}
		}
		obs.id = id;
	}
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html
	for (auto& p : particles) {
		// Predict measurements to all the map landmarks within sensor range for each particle
		std::vector<LandmarkObs> predicted;
		for (auto l : map_landmarks.landmark_list) {
			LandmarkObs predicted_landmark;
			if (dist(p.x, p.y, l.x_f, l.y_f) < sensor_range) {
				predicted_landmark.id = l.id_i;
				predicted_landmark.x = l.x_f;
				predicted_landmark.y = l.y_f;
				predicted.push_back(predicted_landmark);
			}
		}

		if (predicted.size() == 0) {
			// Cannot find any landmark within sensor range
			cout << "predicted size=0\n";
			continue;
		}

		// Transform the vehicle's coordinate system to the map's coordinate system
		std::vector<LandmarkObs> transformed_observations;
		for (auto obs : observations) {
			LandmarkObs landmark;
			landmark.x = p.x + (cos(p.theta) * obs.x) - (sin(p.theta) * obs.y);
			landmark.y = p.y + (sin(p.theta) * obs.x) + (cos(p.theta) * obs.y);
			transformed_observations.push_back(landmark);
		}

		dataAssociation(predicted, transformed_observations);

		// Update the particle's weight using a multi-variate Gaussian distribution
		double prob = 1.0;
		for (auto obs : transformed_observations) {
			double std_x = std_landmark[0];
			double std_y = std_landmark[1];
			double mu_x = map_landmarks.landmark_list[obs.id - 1].x_f;
			double mu_y = map_landmarks.landmark_list[obs.id - 1].y_f;
			double dist = (1 / (2 * M_PI * std_x * std_y)) * exp(-(pow(obs.x - mu_x, 2) / (2 * pow(std_x, 2)) +
					pow(obs.y - mu_y, 2) / (2 * pow(std_y, 2))));
			prob *= dist;
		}
		p.weight = prob;
	}
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution

	weights.clear();
	for (auto& p : particles)
		weights.push_back(p.weight);

	random_device rd;
	mt19937 gen(rd());
	discrete_distribution<> d(weights.begin(), weights.end());

	vector<Particle> picked_particles;
	for (int i = 0; i < particles.size(); i++)
		picked_particles.push_back(particles[d(gen)]);
	particles = picked_particles;
}

Particle ParticleFilter::SetAssociations(Particle particle, std::vector<int> associations, std::vector<double> sense_x, std::vector<double> sense_y)
{
	//particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
	// associations: The landmark id that goes along with each listed association
	// sense_x: the associations x mapping already converted to world coordinates
	// sense_y: the associations y mapping already converted to world coordinates

	//Clear the previous associations
	particle.associations.clear();
	particle.sense_x.clear();
	particle.sense_y.clear();

	particle.associations= associations;
 	particle.sense_x = sense_x;
 	particle.sense_y = sense_y;

 	return particle;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
