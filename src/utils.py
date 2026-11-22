"""
Utility Functions and Recommendation Engine
Student Mental Health Risk Identification and Support System

This module provides:
1. Risk score to ethical categorical tier mapping ("Low", "Moderate", "High").
2. Contextual support recommendation engine based on individual student risk drivers.
3. Crisis detection and immediate emergency support resource routing.
4. Metric evaluation and serialization helpers.
"""

from typing import Dict, Any, List


CRISIS_RESOURCES = {
    "India": {
        "KIRAN (Mental Health Helpline)": "1800-599-0019 (24/7 Toll-Free)",
        "Tele-MANAS": "14416 / 1800-891-4416 (Govt of India 24/7)",
        "AASRA": "+91-9820466726 (24/7 Helpline)",
        "Vandrevala Foundation": "+91-9999 666 555 (24/7 WhatsApp & Call)",
        "Emergency Services": "112"
    },
    "United States": {
        "Suicide & Crisis Lifeline": "Call or Text 988 (24/7)",
        "Crisis Text Line": "Text HOME to 741741",
        "The Trevor Project (LGBTQ+)": "1-866-488-7386 or Text START to 678-678",
        "Emergency Services": "911"
    },
    "United Kingdom": {
        "Samaritans": "Call 116 123 (Free 24/7)",
        "Shout Crisis Text Line": "Text SHOUT to 85258",
        "Emergency Services": "999"
    },
    "International": {
        "Befrienders Worldwide": "https://www.befrienders.org/",
        "Find A Helpline": "https://findahelpline.com/",
        "International Association for Suicide Prevention (IASP)": "https://www.iasp.info/resources/Crisis_Centres/"
    }
}


def map_probability_to_tier(prob: float, threshold_low: float = 0.35, threshold_high: float = 0.65) -> Dict[str, Any]:
    """
    Map model probability score to ethical non-diagnostic risk indicator tiers.
    
    Terminology:
    - Low indicators (< 0.35): Typical academic stress levels; standard student wellness resources.
    - Moderate indicators (0.35 - 0.65): Noticeable stress, sleep debt, or academic strain; proactive counseling suggested.
    - High indicators (> 0.65): Significant cumulative distress indicators; strongly encouraged to consult a professional.
    """
    if prob < threshold_low:
        tier = "Low indicators"
        color = "#28a745"
        badge = "🟢"
        description = "Responses reflect typical academic challenges. Routine wellness and balanced study routines are recommended."
    elif prob <= threshold_high:
        tier = "Moderate indicators"
        color = "#ffc107"
        badge = "🟡"
        description = "Elevated indicators detected across stress, academic load, or sleep. Early check-in with a peer mentor or campus counselor is encouraged."
    else:
        tier = "High indicators"
        color = "#dc3545"
        badge = "🔴"
        description = "Significant cumulative distress indicators detected. We strongly suggest speaking with a university counselor or licensed mental health professional."

    return {
        "tier": tier,
        "score": float(round(prob, 4)),
        "percentage": f"{prob * 100:.1f}%",
        "color": color,
        "badge": badge,
        "description": description
    }


def generate_support_recommendations(student_data: Dict[str, Any], risk_tier: str) -> List[Dict[str, str]]:
    """
    Generate tailored support recommendations based on student's specific input features.
    """
    recs = []

    # 1. Sleep Recommendations
    sleep = student_data.get("Sleep Duration", "")
    if sleep in ["Less than 5 hours", "5-6 hours"]:
        recs.append({
            "category": "Sleep & Physical Health",
            "icon": "🛌",
            "title": "Address Sleep Deprivation",
            "advice": "You reported getting under 6 hours of sleep. Chronic sleep deprivation severely elevates cognitive fatigue and emotional vulnerability. Aim to establish a regular sleep window, reduce screen exposure 45 minutes before bedtime, and consult campus wellness on sleep hygiene."
        })
    elif sleep == "More than 8 hours":
        recs.append({
            "category": "Sleep & Physical Health",
            "icon": "⏰",
            "title": "Evaluate Sleep Quality and Routine",
            "advice": "Extended sleep duration (8+ hours) can sometimes be a sign of fatigue, lethargy, or coping withdrawal. Assess if your sleep feels restorative and consider speaking with a campus health provider if daytime exhaustion persists."
        })

    # 2. Academic Pressure Recommendations
    acad_press = float(student_data.get("Academic Pressure", 0))
    study_hrs = float(student_data.get("Work/Study Hours", 0))
    if acad_press >= 4 or study_hrs >= 9:
        recs.append({
            "category": "Academic Strategy",
            "icon": "📚",
            "title": "Academic Pacing & Workload Optimization",
            "advice": f"You noted high academic pressure ({int(acad_press)}/5) and {study_hrs:.1f} study hours daily. Consider breaking study sessions with the Pomodoro technique (25 min study, 5 min break). Connect with university academic advisors or peer tutoring centers to optimize your course load."
        })

    # 3. Financial Stress
    fin_stress = float(student_data.get("Financial Stress", 0))
    if fin_stress >= 4:
        recs.append({
            "category": "Financial Well-Being",
            "icon": "💰",
            "title": "Campus Financial Aid & Grants",
            "advice": "High financial pressure is one of the leading contributors to university distress. Most universities have emergency micro-grants, work-study openings, and financial counseling offices designed to support students through financial strain."
        })

    # 4. Study Satisfaction
    study_sat = float(student_data.get("Study Satisfaction", 5))
    if study_sat <= 2:
        recs.append({
            "category": "Career & Academic Alignment",
            "icon": "🎯",
            "title": "Course & Career Advising",
            "advice": "Low study satisfaction often points to curriculum misalignment, lack of motivation, or burnout. Scheduling a session with your department academic advisor or career counselor can provide clarity and renewed direction."
        })

    # 5. Dietary Habits
    diet = student_data.get("Dietary Habits", "")
    if diet == "Unhealthy":
        recs.append({
            "category": "Nutrition & Vitality",
            "icon": "🥗",
            "title": "Consistent Nutrition Habits",
            "advice": "Irregular or unhealthy meals directly impact neurotransmitter balance, blood sugar stability, and mood swings. Try planning regular meals, staying hydrated, and keeping simple, nutritious snacks available during study sessions."
        })

    # 6. Family History of Mental Illness
    fam_hist = student_data.get("Family History of Mental Illness", "No")
    if fam_hist == "Yes":
        recs.append({
            "category": "Preventative Mental Health",
            "icon": "🧬",
            "title": "Proactive Mental Wellness Check-in",
            "advice": "Having a family history of mental health challenges means proactive emotional hygiene is especially valuable. Regular mindfulness practices, journaling, and establishing a relationship with a campus counselor before stress peaks can be immensely protective."
        })

    # 7. Overall Tier Guidance
    if risk_tier in ["Moderate indicators", "High indicators"]:
        recs.append({
            "category": "Counseling & Community Support",
            "icon": "🤝",
            "title": "Connect with University Counseling Services",
            "advice": "Reaching out for professional guidance is a strength, not a weakness. University counseling centers offer free, confidential 1-on-1 sessions, support groups, and workshops tailored specifically for student life."
        })
    else:
        recs.append({
            "category": "Maintenance & Growth",
            "icon": "🌱",
            "title": "Sustain Healthy Habits",
            "advice": "Your current responses show balanced stress management. Keep maintaining boundary setting, recreational hobbies, social connections, and regular physical exercise."
        })

    return recs
