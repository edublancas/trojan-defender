"""
Detecting trojans in neural networks
"""

from trojan_defender.detect.detect import create_optimizing_detector, train_optimizing_detector, get_detector_output

__all__ = ['create_optimizing_detector', 'train_optimizing_detector', 'get_detector_output']
