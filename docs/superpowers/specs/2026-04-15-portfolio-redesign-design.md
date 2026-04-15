# Portfolio Website Redesign — Design Spec

**Date:** 2026-04-15
**Author:** Claude (Senior HCI/Frontend perspective)
**Status:** Approved

## Goal

Transform Koyilbek Valiev's portfolio from a functional but rough glassmorphism page into a premium, polished dark-theme portfolio that serves both academic (KAUST IVUL) and industry audiences.

## Audience

Dual: hiring managers at top AI labs/companies + academic researchers/professors.

## Approach

Premium dark redesign — keep dark theme DNA, elevate to Apple-inspired quality with refined typography, categorized skills, proper responsiveness, micro-interactions, and complete valid HTML.

## Design Decisions

### Color Palette
- Background: `#0b0f1a`
- Card background: `rgba(255,255,255,0.03)` with `backdrop-filter: blur(20px)`
- Primary: `#6C63FF` (blue-violet)
- Accent: `#00D4AA` (cyan-green)
- Text primary: `#E8E8E8`
- Text secondary: `#8892B0`
- Borders: `rgba(255,255,255,0.08)`

### Typography
- Headings: Inter 700
- Body: Inter 400/500
- Code/tech: JetBrains Mono 400

### Sections (in order)
1. **Navigation** — fixed, glassmorphic, hamburger on mobile, active section indicator
2. **Hero** — avatar with gradient border glow, name, title, summary, CTA buttons, status badge
3. **Work Experience** — vertical timeline, glass cards, structured bullet points
4. **Skills** — categorized grid with category headers and icons
5. **Education** — glass card, clean layout
6. **Featured Projects** — card grid with tech stack tags
7. **Contact** — responsive info grid + social links with ARIA
8. **Footer** — copyright, quick links, social icons

### New Skills to Add
- **Agentic AI:** Agno, CrewAI, LangGraph, Anthropic Claude API, OpenAI API
- **DevOps:** Docker, Docker Compose, CI/CD, Nginx, Linux
- **Databases:** PostgreSQL, Redis
- **Cloud:** AWS, GCP
- **UI/Demo:** Gradio, Streamlit
- **Experiment Tracking:** W&B (Weights & Biases)

### Technical Constraints
- Single `index.html` with inline CSS/JS (GitHub Pages compatible)
- No build tools or frameworks
- Font Awesome via CDN
- Google Fonts via CDN (Inter + JetBrains Mono)
- Fully valid HTML5
- Mobile-first responsive (480px, 768px, 1024px breakpoints)
- CSS-only animations, Intersection Observer for nav
- All ARIA labels on interactive elements

## Out of Scope
- Multi-page site
- Backend / CMS
- Blog functionality
- Contact form (just info display)
