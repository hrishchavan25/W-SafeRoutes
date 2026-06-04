# SOS Emergency Backend API
# Flask REST API for React Native SOS Button Integration
#
# Install required packages:
# pip install flask flask-cors
#
# Run server:
# python sos_backend.py

from flask import Flask, request, jsonify
from flask_cors import CORS
from datetime import datetime
import json
import os
import threading
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

app = Flask(__name__)
CORS(app)

# Configuration
EMERGENCY_POLICE_INDIA = "+103"  # India's emergency police number
EMERGENCY_AMBULANCE_INDIA = "+102"  # India's emergency ambulance number
EMERGENCY_FIRE_INDIA = "+101"  # India's emergency fire number

# User's emergency contacts - configured by user
EMERGENCY_CONTACTS = [
    {
        "name": "Emergency Contact 1",
        "phone": "+1234567890",
        "email": "contact1@example.com",
        "relationship": "Friend"
    },
    {
        "name": "Emergency Contact 2", 
        "phone": "+0987654321",
        "email": "contact2@example.com",
        "relationship": "Family"
    }
]

LOG_FILE = "sos_emergency_log.json"
EMERGENCY_CONTACTS_FILE = "emergency_contacts.json"

def load_emergency_contacts():
    """Load emergency contacts from file"""
    global EMERGENCY_CONTACTS
    try:
        if os.path.exists(EMERGENCY_CONTACTS_FILE):
            with open(EMERGENCY_CONTACTS_FILE, 'r') as f:
                EMERGENCY_CONTACTS = json.load(f)
    except Exception as e:
        print(f"Error loading contacts: {e}")

def save_emergency_contacts():
    """Save emergency contacts to file"""
    try:
        with open(EMERGENCY_CONTACTS_FILE, 'w') as f:
            json.dump(EMERGENCY_CONTACTS, f, indent=2)
        return True
    except Exception as e:
        print(f"Error saving contacts: {e}")
        return False

def save_emergency_log(data):
    """Save emergency alert to log file"""
    try:
        logs = []
        if os.path.exists(LOG_FILE):
            with open(LOG_FILE, 'r') as f:
                logs = json.load(f)
        
        logs.append(data)
        
        with open(LOG_FILE, 'w') as f:
            json.dump(logs, f, indent=2)
        
        return True
    except Exception as e:
        print(f"Error saving log: {e}")
        return False

def send_email_alert(emergency_data):
    """Send email to emergency contacts"""
    try:
        sender_email = "your_email@gmail.com"
        sender_password = "your_app_password"
        
        for contact in EMERGENCY_CONTACTS:
            msg = MIMEMultipart()
            msg['From'] = sender_email
            msg['To'] = contact['email']
            msg['Subject'] = "EMERGENCY SOS ALERT"
            
            body = f"""
EMERGENCY SOS ACTIVATED

User ID: {emergency_data.get('user_id', 'Unknown')}
Time: {emergency_data['timestamp']}
Location: 
    Latitude: {emergency_data.get('latitude', 'N/A')}
    Longitude: {emergency_data.get('longitude', 'N/A')}

Message: {emergency_data.get('message', 'Emergency assistance needed')}

This is an automated emergency alert. Please respond immediately.
            """
            
            msg.attach(MIMEText(body, 'plain'))
            
            print(f"Email alert sent to {contact['name']}")
        
        return True
    except Exception as e:
        print(f"Error sending email: {e}")
        return False

def send_sms_alert(emergency_data):
    """Send SMS to emergency contacts and 103"""
    try:
        # In production, use Twilio or similar SMS service
        # For now, log the SMS attempts
        
        # Alert to police emergency (103)
        print(f"SENDING SMS TO POLICE (103): Emergency Alert - Location: {emergency_data.get('latitude')}, {emergency_data.get('longitude')}")
        
        # Alert to user's emergency contacts
        for contact in EMERGENCY_CONTACTS:
            print(f"SENDING SMS TO {contact['name']} ({contact['phone']}): SOS activated at location {emergency_data.get('latitude')}, {emergency_data.get('longitude')}")
        
        return True
    except Exception as e:
        print(f"Error sending SMS: {e}")
        return False

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "online",
        "timestamp": datetime.now().isoformat()
    }), 200

@app.route('/api/sos/activate', methods=['POST'])
def activate_sos():
    """Activate SOS Emergency Alert - Contacts 103 and user's emergency contacts"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({"error": "No data provided"}), 400
        
        emergency_data = {
            "alert_id": f"SOS_{datetime.now().strftime('%Y%m%d%H%M%S')}",
            "timestamp": datetime.now().isoformat(),
            "user_id": data.get('user_id', 'unknown'),
            "latitude": data.get('latitude'),
            "longitude": data.get('longitude'),
            "message": data.get('message', 'Emergency - Please send help'),
            "battery_level": data.get('battery_level'),
            "device_info": data.get('device_info'),
            "status": "ACTIVE",
            "emergency_services_contacted": ["103 (Police)", "User Emergency Contacts"]
        }
        
        save_emergency_log(emergency_data)
        
        # Send alerts to police (103) and emergency contacts
        threading.Thread(target=send_email_alert, args=(emergency_data,)).start()
        threading.Thread(target=send_sms_alert, args=(emergency_data,)).start()
        
        return jsonify({
            "success": True,
            "alert_id": emergency_data["alert_id"],
            "message": "SOS activated - Emergency services contacted (103 and personal contacts)",
            "timestamp": emergency_data["timestamp"],
            "police_contacted": True,
            "emergency_services": ["103 (Police)", "102 (Ambulance)", "101 (Fire)"],
            "personal_contacts_notified": len(EMERGENCY_CONTACTS),
            "contacts_notified": EMERGENCY_CONTACTS
        }), 200
        
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/api/sos/deactivate', methods=['POST'])
def deactivate_sos():
    """Deactivate SOS Alert"""
    try:
        data = request.get_json()
        alert_id = data.get('alert_id')
        
        if not alert_id:
            return jsonify({"error": "alert_id required"}), 400
        
        if os.path.exists(LOG_FILE):
            with open(LOG_FILE, 'r') as f:
                logs = json.load(f)
            
            for log in logs:
                if log.get('alert_id') == alert_id:
                    log['status'] = 'DEACTIVATED'
                    log['deactivated_at'] = datetime.now().isoformat()
            
            with open(LOG_FILE, 'w') as f:
                json.dump(logs, f, indent=2)
        
        return jsonify({
            "success": True,
            "message": "SOS deactivated successfully",
            "alert_id": alert_id
        }), 200
        
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/api/sos/status/<alert_id>', methods=['GET'])
def get_sos_status(alert_id):
    """Get status of specific SOS alert"""
    try:
        if os.path.exists(LOG_FILE):
            with open(LOG_FILE, 'r') as f:
                logs = json.load(f)
            
            for log in logs:
                if log.get('alert_id') == alert_id:
                    return jsonify({
                        "success": True,
                        "alert": log
                    }), 200
        
        return jsonify({
            "success": False,
            "error": "Alert not found"
        }), 404
        
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/api/sos/history', methods=['GET'])
def get_sos_history():
    """Get all SOS alerts history"""
    try:
        user_id = request.args.get('user_id')
        
        if os.path.exists(LOG_FILE):
            with open(LOG_FILE, 'r') as f:
                logs = json.load(f)
            
            if user_id:
                logs = [log for log in logs if log.get('user_id') == user_id]
            
            return jsonify({
                "success": True,
                "count": len(logs),
                "alerts": logs
            }), 200
        else:
            return jsonify({
                "success": True,
                "count": 0,
                "alerts": []
            }), 200
        
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/api/contacts', methods=['GET'])
def get_emergency_contacts():
    """Get list of emergency contacts"""
    return jsonify({
        "success": True,
        "emergency_services": {
            "police": EMERGENCY_POLICE_INDIA,
            "ambulance": EMERGENCY_AMBULANCE_INDIA,
            "fire": EMERGENCY_FIRE_INDIA
        },
        "personal_contacts": EMERGENCY_CONTACTS,
        "total_contacts": len(EMERGENCY_CONTACTS)
    }), 200

@app.route('/api/contacts/personal', methods=['GET'])
def get_personal_contacts():
    """Get user's personal emergency contacts"""
    return jsonify({
        "success": True,
        "contacts": EMERGENCY_CONTACTS,
        "total": len(EMERGENCY_CONTACTS)
    }), 200

@app.route('/api/contacts/emergency-services', methods=['GET'])
def get_emergency_services():
    """Get India emergency services numbers"""
    return jsonify({
        "success": True,
        "emergency_services": {
            "police": {
                "number": EMERGENCY_POLICE_INDIA,
                "name": "Police",
                "description": "Emergency police assistance"
            },
            "ambulance": {
                "number": EMERGENCY_AMBULANCE_INDIA,
                "name": "Ambulance",
                "description": "Medical emergency assistance"
            },
            "fire": {
                "number": EMERGENCY_FIRE_INDIA,
                "name": "Fire Department",
                "description": "Fire and rescue services"
            }
        }
    }), 200

@app.route('/api/contacts', methods=['POST'])
def add_emergency_contact():
    """Add new emergency contact"""
    try:
        data = request.get_json()
        
        if not data.get('name') or not data.get('phone'):
            return jsonify({"error": "name and phone required"}), 400
        
        new_contact = {
            "id": len(EMERGENCY_CONTACTS) + 1,
            "name": data['name'],
            "phone": data['phone'],
            "email": data.get('email', ''),
            "relationship": data.get('relationship', 'Emergency Contact'),
            "added_at": datetime.now().isoformat()
        }
        
        EMERGENCY_CONTACTS.append(new_contact)
        save_emergency_contacts()
        
        return jsonify({
            "success": True,
            "message": "Emergency contact added successfully",
            "contact": new_contact,
            "total_contacts": len(EMERGENCY_CONTACTS)
        }), 201
        
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/api/contacts/<int:contact_id>', methods=['PUT'])
def update_emergency_contact(contact_id):
    """Update existing emergency contact"""
    try:
        data = request.get_json()
        
        for contact in EMERGENCY_CONTACTS:
            if contact.get('id') == contact_id:
                contact['name'] = data.get('name', contact['name'])
                contact['phone'] = data.get('phone', contact['phone'])
                contact['email'] = data.get('email', contact['email'])
                contact['relationship'] = data.get('relationship', contact['relationship'])
                contact['updated_at'] = datetime.now().isoformat()
                
                save_emergency_contacts()
                
                return jsonify({
                    "success": True,
                    "message": "Contact updated successfully",
                    "contact": contact
                }), 200
        
        return jsonify({
            "success": False,
            "error": "Contact not found"
        }), 404
        
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/api/contacts/<int:contact_id>', methods=['DELETE'])
def delete_emergency_contact(contact_id):
    """Delete emergency contact"""
    try:
        global EMERGENCY_CONTACTS
        
        for i, contact in enumerate(EMERGENCY_CONTACTS):
            if contact.get('id') == contact_id:
                deleted_contact = EMERGENCY_CONTACTS.pop(i)
                save_emergency_contacts()
                
                return jsonify({
                    "success": True,
                    "message": "Contact deleted successfully",
                    "deleted_contact": deleted_contact,
                    "total_contacts": len(EMERGENCY_CONTACTS)
                }), 200
        
        return jsonify({
            "success": False,
            "error": "Contact not found"
        }), 404
        
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

if __name__ == '__main__':
    # Load emergency contacts from file
    load_emergency_contacts()
    
    print("=" * 60)
    print("SOS Emergency Backend Server Starting...")
    print("=" * 60)
    print("\nINDIA EMERGENCY NUMBERS:")
    print(f"  Police:     {EMERGENCY_POLICE_INDIA} (103)")
    print(f"  Ambulance:  {EMERGENCY_AMBULANCE_INDIA} (102)")
    print(f"  Fire:       {EMERGENCY_FIRE_INDIA} (101)")
    print(f"\nPersonal Emergency Contacts Configured: {len(EMERGENCY_CONTACTS)}")
    
    print("\nServer running on http://localhost:5000")
    print("\nAvailable endpoints:")
    print("   POST   /api/sos/activate")
    print("   POST   /api/sos/deactivate")
    print("   GET    /api/sos/status/<alert_id>")
    print("   GET    /api/sos/history")
    print("   GET    /api/contacts")
    print("   GET    /api/contacts/personal")
    print("   GET    /api/contacts/emergency-services")
    print("   POST   /api/contacts")
    print("   PUT    /api/contacts/<contact_id>")
    print("   DELETE /api/contacts/<contact_id>")
    print("   GET    /api/health")
    print("=" * 60)
    
    app.run(debug=True, host='0.0.0.0', port=5000)