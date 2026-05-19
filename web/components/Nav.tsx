"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";

const ROUTES = [
  { href: "/", label: "Triage" },
  { href: "/explainability", label: "Explainability" },
  { href: "/search", label: "Semantic Search" },
];

export default function Nav() {
  const path = usePathname();
  return (
    <div className="topbar">
      <span className="brand">◗ Radiology AI</span>
      <nav className="modnav">
        {ROUTES.map((r) => (
          <Link key={r.href} href={r.href}
            className={path === r.href ? "active" : ""}>
            {r.label}
          </Link>
        ))}
      </nav>
      <span className="corpus">NIH ChestX-ray14 · DenseNet121 · CLIP ViT-B/32</span>
    </div>
  );
}
