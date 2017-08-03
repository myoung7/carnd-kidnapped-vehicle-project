

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

const int NUM_OF_PARTICLES = 50;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
    
    default_random_engine gen;
    
    weights = vector<double>(NUM_OF_PARTICLES);
    
    for (int i=0; i<NUM_OF_PARTICLES; i++) {
        const double std_x     = std[0];
        const double std_y     = std[1];
        const double std_theta = std[2];
        
        normal_distribution<double> dist_x(x, std_x);
        normal_distribution<double> dist_y(y, std_y);
        normal_distribution<double> dist_theta(theta, std_theta);
        
        Particle particle;
        particle.id     = i;
        particle.x      = dist_x(gen);
        particle.y      = dist_y(gen);
        particle.theta  = dist_theta(gen);
        particle.weight = 1;
        
        particles.push_back(particle);
    }
    
    is_initialized = true;
    
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
    
    default_random_engine gen;
    
    normal_distribution<double> dist_x(0, std_pos[0]);
    normal_distribution<double> dist_y(0, std_pos[1]);
    normal_distribution<double> dist_theta(0, std_pos[2]);
    
    for (int i=0; i<NUM_OF_PARTICLES; i++) {
        
        if (yaw_rate == 0) {
            particles[i].x += velocity * delta_t * cos(particles[i].theta);
            particles[i].y += velocity * delta_t * sin(particles[i].theta);
        } else {
            particles[i].x += velocity/yaw_rate*(sin(particles[i].theta
                                                     + yaw_rate*delta_t) - sin(particles[i].theta));
            particles[i].y += velocity/yaw_rate*(cos(particles[i].theta)
                                                 - cos(particles[i].theta + yaw_rate*delta_t));
            particles[i].theta += yaw_rate*delta_t;
        }
        
        particles[i].x     += dist_x(gen);
        particles[i].y     += dist_y(gen);
        particles[i].theta += dist_theta(gen);
    }
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
    // TODO: Find the predicted measurement that is closest to each observed measurement and assign the
    //   observed measurement to this particular landmark.
    // NOTE: this method will NOT be called by the grading code. But you will probably find it useful to
    //   implement this method and use it as a helper during the updateWeights phase.
    
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
                                   std::vector<LandmarkObs> observations, Map map_landmarks) {
    
    std::vector<LandmarkObs> predictions;
    
    const double std_l_x = std_landmark[0];
    const double std_l_y = std_landmark[1];
    
    const double coef = 1/(2.0*M_PI*std_l_x*std_l_y);
    
    double weights_sum = 0;
    
    for (int i = 0; i < particles.size(); i++) {
        
        const double part_x = particles[i].x;
        const double part_y = particles[i].y;
        const double part_theta = particles[i].theta;
        
        double currentWeight = 1;
        
        for (int l=0; l<observations.size(); l++) {
            LandmarkObs translatedObservation;
            const double obs_x     = observations[l].x;
            const double obs_y     = observations[l].y;
            
            const double new_x = obs_x * cos(part_theta)
                                 - obs_y * sin(part_theta) + part_x;
            const double new_y = obs_x * sin(part_theta)
                                 + obs_y * cos(part_theta) + part_y;
            
            translatedObservation.x = new_x;
            translatedObservation.y = new_y;
            
            double minimum_distance = 10000000;
            
            for (int j=0; j<map_landmarks.landmark_list.size(); j++) {
                Map::single_landmark_s currentLandmark = map_landmarks.landmark_list[j];
                const double predicted_landmark_x = currentLandmark.x_f;
                const double predicted_landmark_y = currentLandmark.y_f;
                
                const double distance = dist(translatedObservation.x, translatedObservation.y,
                                       currentLandmark.x_f, currentLandmark.y_f);
                
                if (distance < minimum_distance) {
                    minimum_distance = distance;
                    translatedObservation.id = currentLandmark.id_i;
                }
            }
            
            for (int k=0; k<map_landmarks.landmark_list.size(); k++) {
                
                Map::single_landmark_s currentLandmark = map_landmarks.landmark_list[k];
                
                if ( translatedObservation.id == currentLandmark.id_i ) {
                    
                    const double o_x = translatedObservation.x;
                    const double o_y = translatedObservation.y;
                    
                    const double l_x = currentLandmark.x_f;
                    const double l_y = currentLandmark.y_f;
                    
                    const double e_term = exp(-1.0/2 * ((pow(o_x-l_x,2)/(pow(std_l_x,2)))
                                                  + (pow(o_y-l_y,2)/(pow(std_l_y,2)))));
                    currentWeight *= coef * e_term;
                }
                
            }
        }
        
        weights[i] = currentWeight;
        particles[i].weight = currentWeight;
        weights_sum += currentWeight;
        
    }
}

void ParticleFilter::resample() {
    
    //Source: https://discussions.udacity.com/t/output-always-zero/260432/11?
    
    vector<Particle> new_particles;
    
    default_random_engine gen;
    discrete_distribution<int> d(weights.begin(), weights.end());
    
    for (int i=0; i<NUM_OF_PARTICLES; i++) {
        
        Particle new_particle = particles[d(gen)];
        new_particles.push_back(new_particle);
        
    }
    
    particles = new_particles;
    
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
