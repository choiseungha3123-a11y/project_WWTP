package kr.kro.prjectwwtp.persistence;

import org.springframework.data.jpa.repository.JpaRepository;

import kr.kro.prjectwwtp.domain.MemoLog;

public interface MemoLogRepository extends JpaRepository<MemoLog, Long>{

}