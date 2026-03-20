#!/bin/bash
echo "Starting Security Audit of Dependencies..."
pip audit
safety check -r requirements.txt
